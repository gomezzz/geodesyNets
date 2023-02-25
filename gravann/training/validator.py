import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gravann.functions import integration
from gravann.labels import mascon, polyhedral, calculate_density
from gravann.util import fixRandomSeeds
from gravann.util import get_target_point_sampler
from .kappa import compute_c_for_model
from .losses import contrastive_loss, normalized_L1_loss, normalized_relative_L2_loss, \
    normalized_relative_component_loss, RMSE, relRMSE


def validate(model, encoding, sample, ground_truth, use_acc=True, N_integration=500000, **kwargs):
    """Convenience function to compute the different loss values for the passed model and asteroid with high precision

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        sample (str): the name of the asteroid (also used as filepath)
        ground_truth (str): 'polyhedral' or 'mascon' or 'polyhedral-mascon' (polyhedral as groundtruth to mascon model)
        use_acc (bool, optional): if to use the acceleration labels instead of the potential
        N_integration (int, optional): Number of integrations points to use. Defaults to 500000.
        **kwargs: see below

    Keyword Args:
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): non-uniform asteroid masses. Pass if using differential training
        mesh_vertices ((N, 3) array): mesh vertices
        mesh_faces ((M, 3) array): mesh triangles
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df

    """
    if ground_truth == 'mascon':
        label_function, prediction_function = _validation_mascon(model, encoding, use_acc, N_integration, **kwargs)
        return _validation(label_function, prediction_function, sample, **kwargs)
    elif ground_truth == 'polyhedral':
        label_function, prediction_function = _validation_polyhedral(model, encoding, use_acc, N_integration, **kwargs)
        return _validation(label_function, prediction_function, sample, **kwargs)
    elif ground_truth == 'polyhedral-mascon':
        # Model is not required and can be None in these cases
        mascon_label, _ = _validation_mascon(model, encoding, use_acc, N_integration, **kwargs)
        polyhedral_label, _ = _validation_polyhedral(model, encoding, use_acc, N_integration, **kwargs)
        return _validation(polyhedral_label, mascon_label, sample, **kwargs)
    else:
        raise NotImplementedError(f"The method {ground_truth} is not implemented!")


def _validation_mascon(model, encoding, use_acc, N_integration, **kwargs):
    """Generates the label_function and the prediction function for the mascon model
    """
    mascon_points, mascon_masses, = kwargs['mascon_points'], kwargs['mascon_masses']
    mascon_masses_nu = kwargs.get('mascon_masses_nu', None)
    integration_grid, h, N_int = integration.compute_integration_grid(N_integration)

    def prediction_adjustment(tp, mp, mm, x):
        return x

    if use_acc:
        label_function = mascon.acceleration
        integrator = integration.acceleration_trapezoid
    else:
        label_function = mascon.potential
        integrator = integration.potential_trapezoid
    if mascon_masses_nu is not None:
        c = compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc)

        # Labels for differential need to be computed on non-uniform ground truth
        def label_function(tp, mp, mm):
            return mascon.acceleration(tp, mp, mascon_masses_nu)

        # Predictions for differential need to be adjusted with acceleration from uniform ground truth
        def prediction_adjustment(tp, mp, mm, x):
            return mascon.acceleration(tp, mp, mm) + c * x
    return (
        lambda points: label_function(points, mascon_points, mascon_masses),
        lambda points: prediction_adjustment(points, mascon_points, mascon_masses,
                                             integrator(points, model, encoding, N=N_int,
                                                        h=h, sample_points=integration_grid))
    )


def _validation_polyhedral(model, encoding, use_acc, N_integration, **kwargs):
    """Generates the label_function and the prediction function for the polyhedral model
    """
    mesh_vertices, mesh_faces, = kwargs['mesh_vertices'], kwargs['mesh_faces']
    density = calculate_density(mesh_vertices, mesh_faces)
    integration_grid, h, N_int = integration.compute_integration_grid(N_integration)
    if use_acc:
        label_function = polyhedral.acceleration
        integrator = integration.acceleration_trapezoid
    else:
        label_function = polyhedral.potential
        integrator = integration.potential_trapezoid
    return (
        lambda points: label_function(points, mesh_vertices, mesh_faces, density),
        lambda points: integrator(points, model, encoding, N=N_int, h=h, sample_points=integration_grid)
    )


def _validation(label_function, prediction_function, sample, **kwargs):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        label_function: the original training labels
        prediction_function: the prediction function of the trained model
        sample (str): name of the body (equals file name)

    Keyword Args:
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df
    """

    # Default arguments for the Keyword Args
    N = kwargs.get("N", 5000)
    sampling_altitudes = kwargs.get("sampling_altitudes", [0.05, 0.1, 0.25])
    batch_size = kwargs.get("batch_size", 100)
    russell_points = kwargs.get("russell_points", 3)
    progressbar = kwargs.get("progressbar", True)

    torch.cuda.empty_cache()
    fixRandomSeeds()

    asteroid_pk_path = f"./3dmeshes/{sample}.pk"

    loss_fns = [normalized_L1_loss, normalized_relative_component_loss, RMSE, relRMSE]
    cols = ["Altitude", "Normalized L1 Loss", "Normalized Relative Component Loss", "RMSE", "relRMSE"]
    results = pd.DataFrame(columns=cols)

    ###############################################
    # Compute validation for radially projected points (outside the asteroid),

    # Low altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(russell_points, method="radial_projection", bounds=[
        0.0, 0.15625], limit_shape_to_asteroid=asteroid_pk_path)

    target_points = target_sampler().detach()

    if progressbar:
        pbar = tqdm(desc="Computing validation...",
                    total=2 * len(target_points) + N * (len(sampling_altitudes)))

    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(range(idx * batch_size,
                             np.minimum((idx + 1) * batch_size, len(target_points))))
        points = target_points[indices]
        labels.append(label_function(points).detach())
        prediction = prediction_function(points).detach()
        pred.append(prediction)
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                       relRMSE]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(torch.mean(
                loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

    results = results.append(
        dict(zip(cols, ["Low Altitude"] + loss_values)), ignore_index=True)

    # High altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(russell_points, method="radial_projection", bounds=[
        0.15625, 0.3125], limit_shape_to_asteroid=asteroid_pk_path)

    target_points = target_sampler().detach()
    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(range(idx * batch_size,
                             np.minimum((idx + 1) * batch_size, len(target_points))))
        points = target_points[indices]
        labels.append(label_function(points).detach())
        prediction = prediction_function(points).detach()
        pred.append(prediction)
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                       relRMSE]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(torch.mean(
                loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

    results = results.append(
        dict(zip(cols, ["High Altitude"] + loss_values)), ignore_index=True)

    ################################################
    # Compute errors at different altitudes
    for idx, altitude in enumerate(sampling_altitudes):
        torch.cuda.empty_cache()
        pred, labels, loss_values = [], [], []
        target_sampler = get_target_point_sampler(
            N=batch_size, method="altitude",
            bounds=[altitude], limit_shape_to_asteroid=asteroid_pk_path)
        for batch in range(N // batch_size):
            target_points = target_sampler().detach()
            labels.append(label_function(target_points).detach())

            prediction = prediction_function(target_points).detach()
            pred.append(prediction)

            if progressbar:
                pbar.update(batch_size)
            torch.cuda.empty_cache()
        pred = torch.cat(pred)
        labels = torch.cat(labels)

        # Compute Losses
        for loss_fn in loss_fns:
            if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss, RMSE,
                           relRMSE]:
                loss_values.append(torch.mean(
                    loss_fn(pred, labels)).cpu().detach().item())
            else:
                loss_values.append(torch.mean(
                    loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

        results = results.append(
            dict(zip(cols, ["Altitude_" + str(idx)] + loss_values)), ignore_index=True)

    if progressbar:
        pbar.close()

    torch.cuda.empty_cache()
    return results


def validation_results_unpack_df(validation_results):
    """Converts validation df to data row

    Args:
        validation_results (pandas.df): validation results

    Returns:
        pandas.df: df as one row
    """
    v = validation_results.set_index("Altitude")
    v = v.unstack().to_frame().sort_index(level=1).T
    v.columns = [x + '@' + str(y) for (x, y) in v.columns]
    return v
