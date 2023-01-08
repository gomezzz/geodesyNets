import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gravann.polyhedral import ACC_L as POLYHEDRAL_ACC_L, U_L as POLYHEDRAL_U_L, calculate_density
from ._integration import ACC_trap, U_trap_opt, compute_integration_grid
from ._io import load_sample, load_polyhedral_mesh
from ._losses import contrastive_loss, normalized_L1_loss, normalized_relative_L2_loss, \
    normalized_relative_component_loss, RMSE, relRMSE
from ._mascon_labels import ACC_L as MASCON_ACC_L, U_L as MASCON_U_L, ACC_L_differential as MASCON_ACC_L_differential
from ._sample_observation_points import get_target_point_sampler
from ._utils import fixRandomSeeds


def compute_c_for_model_v2(model, encoding, method, use_acc=True, **kwargs):
    """Convenience function to calculate the current c constant for a model (based on mascon or polyhedral mesh data)

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        method (str): either 'mascon' or 'polyhedral'
        use_acc (bool): if acceleration should be used (otherwise potential)


    Keyword Args:
        sample (str): sample name (file name) instead of mascons or mesh data
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): asteroid mascon masses
        mesh_vertices ((N, 3) array-like): the vertices of a polyhedron
        mesh_faces ((N, 3) array-like): the triangular faces of a polyhedron
        density (float): the density of a polyhedron
    """
    sample = kwargs.get("sample", None)
    if method == 'mascon':
        if sample is not None:
            mascon_points, mascon_masses, mascon_masses_nu = load_sample(sample, use_acc)
        else:
            mascon_points, mascon_masses, mascon_masses_nu = kwargs.get("mascon_points", None), kwargs.get(
                "mascon_masses", None), kwargs.get("mascon_masses_nu", None)
        return compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc)
    elif method == 'polyhedral':
        if sample is not None:
            mesh_vertices, mesh_faces = load_polyhedral_mesh(sample)
        else:
            mesh_vertices, mesh_faces = kwargs.get("mesh_vertices", None), kwargs.get("mesh_faces", None)
        density = kwargs.get("density", 1.0)
        return _compute_c_for_model_polyhedral(model, encoding, mesh_vertices, mesh_faces, density, use_acc)
    else:
        raise NotImplemented(f"The method {method} is not implemented for compute_c!")


def compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu=None, use_acc=True):
    """Computes the current c constant for a model (given mascon points)

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
    """
    if use_acc:
        if mascon_masses_nu is None:
            return _compute_c_for_model(
                lambda x: MASCON_ACC_L(x, mascon_points, mascon_masses),
                lambda x: ACC_trap(x, model, encoding, N=100000)
            )
        else:
            return _compute_c_for_model(
                lambda x: MASCON_ACC_L_differential(x, mascon_points, mascon_masses, mascon_masses_nu),
                lambda x: ACC_trap(x, model, encoding, N=100000)
            )
    else:
        return _compute_c_for_model(
            lambda x: MASCON_U_L(x, mascon_points, mascon_masses),
            lambda x: U_trap_opt(x, model, encoding, N=100000)
        )


def _compute_c_for_model_polyhedral(model, encoding, mesh_vertices, mesh_faces, density, use_acc=True):
    """Computes the current c constant for a model.

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mesh_vertices ((N, 3) array): mesh vertices
        mesh_faces ((M, 3) array): mesh triangles
        density (Float): the density of the polyhedron
        use_acc (bool): if acceleration should be used (otherwise potential)
    """
    if use_acc:
        return _compute_c_for_model(
            lambda x: POLYHEDRAL_ACC_L(x, mesh_vertices, mesh_faces, density),
            lambda x: ACC_trap(x, model, encoding, N=100000)
        )
    else:
        return _compute_c_for_model(
            lambda x: POLYHEDRAL_U_L(x, mesh_vertices, mesh_faces, density),
            lambda x: U_trap_opt(x, model, encoding, N=100000)
        )


def _compute_c_for_model(label, label_trap):
    """Common method computing the current c constant for a model.

    Args:
        label: mascon or polyhedral label (only takes target points as argument)
        label_trap : label of the trained model (only takes target points as argument)
    """
    targets_point_sampler = get_target_point_sampler(1000, method="spherical", bounds=[0.81, 1.0])
    target_points = targets_point_sampler()
    labels = label(target_points)
    predicted = label_trap(target_points)
    return (torch.sum(predicted * labels) / torch.sum(predicted * predicted)).item()


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


def validation_v2(model, encoding, sample, method, use_acc=True, N_integration=500000, **kwargs):
    """Convenience function to compute the different loss values for the passed model and asteroid with high precision

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        sample (str): the name of the asteroid (also used as filepath)
        method (str): 'polyhedral' or 'mascon'
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
    if method == 'mascon':
        label_function, prediction_function = _validation_mascon(model, encoding, use_acc, N_integration, **kwargs)
        return _validation(label_function, prediction_function, sample, **kwargs)
    elif method == 'polyhedral':
        label_function, prediction_function = _validation_polyhedral(model, encoding, use_acc, N_integration, **kwargs)
        return _validation(label_function, prediction_function, sample, **kwargs)
    else:
        raise NotImplementedError(f"The method {sample} is not implemented!")


def _validation_mascon(model, encoding, use_acc, N_integration, **kwargs):
    """Generates the label_function and the prediction function for the mascon model
    """
    mascon_points, mascon_masses, = kwargs['mascon_points'], kwargs['mascon_masses']
    mascon_masses_nu = kwargs.get('mascon_masses_nu', None)
    integration_grid, h, N_int = compute_integration_grid(N_integration)

    def prediction_adjustment(tp, mp, mm, x):
        return x

    if use_acc:
        label_function = MASCON_ACC_L
        integrator = ACC_trap
    else:
        label_function = MASCON_U_L
        integrator = U_trap_opt
    if mascon_masses_nu is not None:
        c = compute_c_for_model(model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc)

        # Labels for differential need to be computed on non-uniform ground truth
        def label_function(tp, mp, mm):
            return MASCON_ACC_L(tp, mp, mascon_masses_nu)

        # Predictions for differential need to be adjusted with acceleration from uniform ground truth
        def prediction_adjustment(tp, mp, mm, x):
            return MASCON_ACC_L(tp, mp, mm) + c * x
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
    integration_grid, h, N_int = compute_integration_grid(N_integration)
    if use_acc:
        label_function = POLYHEDRAL_ACC_L
        integrator = ACC_trap
    else:
        label_function = POLYHEDRAL_U_L
        integrator = U_trap_opt
    return (
        lambda points: label_function(points, mesh_vertices, mesh_faces, density),
        lambda points: integrator(points, model, encoding, N=N_int, h=h, sample_points=integration_grid)
    )


def _validation(label_function, prediction_function, sample,
                N=5000, sampling_altitudes=[0.05, 0.1, 0.25],
                batch_size=100, russell_points=3, progressbar=True, **kwargs):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        label_function: the original training labels
        prediction_function: the prediction function of the trained model
        sample (str): name of the body (equals file name)
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df
    """
    torch.cuda.empty_cache()
    fixRandomSeeds()

    asteroid_pk_path = f"./3dmeshes/{sample}.pk"

    loss_fns = [normalized_L1_loss,  # normalized_loss, normalized_relative_L2_loss,
                normalized_relative_component_loss, RMSE, relRMSE]
    cols = ["Altitude", "Normalized L1 Loss",  # "Normalized Loss", "Normalized Rel. L2 Loss",
            "Normalized Relative Component Loss", "RMSE", "relRMSE"]
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


def validation(model, encoding, mascon_points, mascon_masses,
               use_acc, asteroid_pk_path, mascon_masses_nu=None,
               N=5000, N_integration=500000, sampling_altitudes=[0.05, 0.1, 0.25],
               batch_size=100, russell_points=3, progressbar=True):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
        asteroid_pk_path (str): path to the asteroid mesh, necessary for altitude
        mascon_masses_nu (torch.tensor): non-uniform asteroid masses. Pass if using differential training
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        N_integration (int, optional): Number of integrations points to use. Defaults to 500000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df
    """
    torch.cuda.empty_cache()
    fixRandomSeeds()

    # identity for non-differential
    def prediction_adjustment(tp, mp, mm, x):
        return x

    if use_acc:
        label_function = MASCON_ACC_L
        integrator = ACC_trap
        integration_grid, h, N_int = compute_integration_grid(N_integration)
    else:
        label_function = MASCON_U_L
        integrator = U_trap_opt
        integration_grid, h, N_int = compute_integration_grid(N_integration)
    if mascon_masses_nu is not None:
        c = compute_c_for_model(
            model, encoding, mascon_points, mascon_masses, mascon_masses_nu, use_acc=use_acc)

        # Labels for differential need to be computed on non-uniform ground truth
        def label_function(tp, mp, mm):
            return MASCON_ACC_L(tp, mp, mascon_masses_nu)

        # Predictions for differential need to be adjusted with acceleration from uniform ground truth
        def prediction_adjustment(tp, mp, mm, x):
            return MASCON_ACC_L(tp, mp, mm) + c * x

    loss_fns = [normalized_L1_loss,  # normalized_loss, normalized_relative_L2_loss,
                normalized_relative_component_loss, RMSE, relRMSE]
    cols = ["Altitude", "Normalized L1 Loss",  # "Normalized Loss", "Normalized Rel. L2 Loss",
            "Normalized Relative Component Loss", "RMSE", "relRMSE"]
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
        labels.append(label_function(
            points, mascon_points, mascon_masses).detach())
        prediction = integrator(points, model, encoding, N=N_int,
                                h=h, sample_points=integration_grid).detach()
        prediction = prediction_adjustment(
            points, mascon_points, mascon_masses, prediction)
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
        labels.append(label_function(
            points, mascon_points, mascon_masses).detach())
        prediction = integrator(points, model, encoding, N=N_int,
                                h=h, sample_points=integration_grid).detach()
        prediction = prediction_adjustment(
            points, mascon_points, mascon_masses, prediction)
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
            labels.append(label_function(
                target_points, mascon_points, mascon_masses).detach())

            prediction = integrator(target_points, model, encoding, N=N_int,
                                    h=h, sample_points=integration_grid).detach()
            prediction = prediction_adjustment(
                target_points, mascon_points, mascon_masses, prediction)
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
