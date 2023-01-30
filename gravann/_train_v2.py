import pathlib
import pickle as pk
import time
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from gravann.polyhedral import ACC_L as POLYHEDRAL_ACC_L
from . import load_polyhedral_mesh, load_mascon_data
# Required for loading runs
from ._encodings import *
from ._io import save_results, save_plots
from ._losses import contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss
# Required for loading runs
from ._mascon_labels import ACC_L as MASCON_ACC_L
from ._plots import plot_model_rejection, plot_model_vs_mascon_contours
from ._sample_observation_points import get_target_point_sampler
from ._train import init_network
from ._utils import fixRandomSeeds, EarlyStopping
from ._validation import validation_results_unpack_df
from ._validation_v2 import validation_v2


def _init_environment(parameters: {str, any}) -> str:
    """Creates the environment (the run-folder) for the given training with parameters

    Args:
        parameters: dictionary of parameters of this training run

    Returns:
        the run folder

    """
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Fix the random seeds for this run
    fixRandomSeeds()

    domain = str(parameters['sample_domain']) \
        .replace('.', '_').replace('[', '').replace(']', '').replace(',', '').replace(' ', '=')

    # Create folder for this specific run
    run_folder = f"""
        {parameters['output_folder']}/
        {parameters['method']}/
        {parameters['sample']}/
        LR={parameters['learning_rate']}_loss={parameters['parameters'].__name__}_ENC={parameters['encoding'].name}_
        BS={parameters['batch_size']}_layers={parameters['hidden_layers']}_neurons={parameters['n_neurons']}_
        METHOD={parameters['target_sample_method']}_DOMAIN={domain}/
        """
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)
    return run_folder


def _init_model_and_optimizer(run_folder: str, encoding: any, n_neurons: int, activation: any, model_type: str,
                              omega: float, hidden_layers: int, learning_rate: float):
    """Initializes the model and the associated training utility

    Args:
        run_folder: the folder where to store the model in case of early stopping
        encoding: encoding instance to use for the network
        n_neurons: the number of neurons per layer
        activation: activation function for the last network layer
        model_type: the model type
        omega: Omega value for siren activations
        hidden_layers: the number of hidden layer
        learning_rate: the utilized learning rate for the optimizer

    Returns:
        Tuple of model, the early_stopper, optimizer, scheduler

    """
    # Initializes the model
    model = init_network(
        encoding,
        n_neurons=n_neurons,
        activation=activation,
        model_type=model_type,
        siren_omega=omega,
        hidden_layers=hidden_layers
    )

    # Initializes training utility: Early Stopping
    early_stopper = EarlyStopping(
        save_folder=run_folder
    )
    # Initializes training utility: Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    # Initializes training utility: Scheduler for Learning Rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=200,
        min_lr=1e-6,
        verbose=True
    )

    return model, early_stopper, optimizer, scheduler


def _init_training_sampler(sample: str, target_sample_method: str, sample_domain: [float], batch_size: int):
    """Creates a new target point sample with the given method and sample domain.

    Args:
        sample: the sample body's name
        target_sample_method: the sample method (e.g. 'spherical')
        sample_domain: the sample domain, specifies the sampling radius.
        batch_size: the number of points per function call

    Returns:
        sampling function

    """
    return get_target_point_sampler(
        batch_size,
        method=target_sample_method,
        bounds=sample_domain,
        limit_shape_to_asteroid=f"f3dmeshes/{sample}_lp.pk"
    )


def _train_on_batch_v2(points, prediction_fn, labels, loss_fn, optimizer, scheduler):
    """Trains the passed model on the passed batch

    Args:
        points (tensor): target points for training
        prediction_fn (func): prediction func of the model
        labels (tensor): labels at the target points
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use

    Returns:
        torch tensor: losses
    """
    predicted = prediction_fn(points)
    c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)

    if loss_fn in [contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss]:
        loss = loss_fn(predicted, labels)
    else:
        loss = loss_fn(predicted.view(-1), labels.view(-1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step(loss.item())

    return loss, c


def _init_prediction_label(integrator, model, encoding, N, integration_domain):
    return lambda points: integrator(points, model, encoding, N=N, domain=integration_domain)


def _init_ground_truth_labels(method, sample):
    if method == 'polyhedral':
        mesh_vertices, mesh_edges = load_polyhedral_mesh(sample)
        return _init_polyhedral_label(mesh_vertices, mesh_edges)
    elif method == 'mascon':
        mascon_points, mascon_masses_u = load_mascon_data(sample)
        return _init_polyhedral_label(mascon_points, mascon_masses_u)
    else:
        raise NotImplemented(f"The method {method} is not implemented!")


def _init_polyhedral_label(mesh_vertices, mesh_edges):
    return lambda points: POLYHEDRAL_ACC_L(points, mesh_vertices, mesh_edges)


def _init_mascon_label(mascon_points, mascon_masses):
    return lambda points: MASCON_ACC_L(points, mascon_points, mascon_masses)


def run_training_v2(cfg: {str, any}):
    """Runs a specific parameter configuration.

    Args:
        cfg: dictionary with the configuration for the training run

    Returns:
        results
    """
    # Time Measurement
    start_time = time.time()

    # Initialize the environment and prepare the run folder
    run_folder = _init_environment(cfg)
    # Initialize the model and associated torch training utility
    model, early_stopper, optimizer, scheduler = _init_model_and_optimizer(
        run_folder=run_folder, encoding=cfg["encoding"], n_neurons=cfg["n_neurons"], activation=cfg["activation"],
        model_type=cfg["model_type"], omega=cfg["omega"], hidden_layers=cfg["hidden_layers"],
        learning_rate=cfg["learning_rate"]
    )
    # Initialize the target point sampler
    target_points_sampler = _init_training_sampler(
        sample=cfg["sample"], target_sample_method=cfg["sample_method"],
        sample_domain=cfg["sample_domain"], batch_size=cfg["batch_size"]
    )
    # Initialize the prediction function by binding the model and defined configuration parameters
    prediction_fn = _init_prediction_label(
        integrator=cfg["integrator"], model=model, encoding=cfg["encoding"],
        N=cfg["N"], integration_domain=cfg["integration_domain"]
    )
    # Initialize the label function by binding the sample data
    label_fn = _init_ground_truth_labels(
        method=cfg["method"], sample=cfg["sample"]
    )

    # When a new network is created we init empty training logs and we init some loss trend indicators
    loss_log, lr_log, weighted_average_log, n_inferences = [], [], [], []
    weighted_average = deque([], maxlen=20)
    target_points, labels = [], []

    t = tqdm(range(cfg["training"]["iterations"]), ncols=150)
    # At the beginning (first plots) we assume no learned c
    c = 1.
    for it in t:
        # Each hundred epochs we produce the plots
        if it % 100 == 0:
            # Save a plot
            plot_model_rejection(
                model, encoding, views_2d=True, bw=True, N=cfg["plotting_points"], alpha=0.1,
                s=50, save_path=run_folder + "rejection_plot_iter" + format(it, '06d') + ".png", c=c,
                progressbar=False
            )
            # TODO Replace
            plot_model_vs_mascon_contours(
                model, encoding, mascon_points, N=cfg["plotting_points"],
                save_path=run_folder + "contour_plot_iter" + format(it, '06d') + ".png", c=c
            )
            plt.close('all')
        # Each ten epochs we resample the target points
        if it % 10 == 0:
            target_points = target_points_sampler()
            labels = label_fn(target_points)

        # Train
        loss, c = _train_on_batch_v2(
            target_points, prediction_fn, labels,
            cfg["loss_fn"], optimizer, scheduler
        )

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        lr_log.append(optimizer.param_groups[0]['lr'])
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((cfg["integration"]["points"] * batch_size) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(f"L={loss.item():.3e} | AvgL={wa_out:.3e} | c={c:.3e}")

        if early_stopper.early_stop(loss.item(), model):
            print(f"Early stopping at minimal loss {early_stopper.minimal_loss}")
            break

        torch.cuda.empty_cache()
    # Restore best checkpoint
    print("Restoring best checkpoint for validation...")
    model.load_state_dict(torch.load(run_folder + "best_model.mdl"))
    # TODO Replace
    validation_results = validation_v2(
        model, encoding, mascon_points, mascon_masses_u,
        cfg["model"]["use_acceleration"], "3dmeshes/" + sample,
        mascon_masses_nu=mascon_masses_nu,
        N_integration=500000, N=cfg["training"]["validation_points"])

    # TODO Tidy up
    save_results(loss_log, weighted_average_log,
                 validation_results, model, run_folder)

    save_plots(model, encoding, mascon_points, lr_log, loss_log,
               weighted_average_log, vision_loss_log, n_inferences, run_folder, c, cfg["plotting_points"])

    # store run config
    cfg_dict = {"Sample": sample, "Type": "ACC" if cfg["model"]["use_acceleration"] else "U",
                "Model": cfg["model"]["type"], "Loss": loss_fn.__name__, "Encoding": encoding.name,
                "Integrator": cfg["integrator"].__name__, "Activation": str(activation)[:-2],
                "n_neurons": cfg["model"]["n_neurons"], "hidden_layers": cfg["model"]["hidden_layers"],
                "Batch Size": batch_size, "LR": cfg["training"]["lr"], "Target Sampler": target_sample_method,
                "Integration Points": cfg["integration"]["points"], "Vision Loss": cfg["training"]["visual_loss"],
                "c": c}

    with open(run_folder + 'config.pk', 'wb') as handle:
        pk.dump(cfg_dict, handle)

    # Compute validation results
    val_res = validation_results_unpack_df(validation_results)

    # Time Measurements
    end_time = time.time()
    runtime = end_time - start_time

    result_dictionary = {"Sample": sample,
                         "Type": "ACC" if cfg["model"]["use_acceleration"] else "U", "Model": cfg["model"]["type"],
                         "Loss": loss_fn.__name__, "Encoding": encoding.name,
                         "Integrator": cfg["integrator"].__name__, "Activation": str(activation)[:-2],
                         "Batch Size": batch_size, "LR": cfg["training"]["lr"], "Target Sampler": target_sample_method,
                         "Integration Points": cfg["integration"]["points"],
                         "Runtime": runtime, "Final Loss": loss_log[-1],
                         "Final WeightedAvg Loss": weighted_average_log[-1], "Final Vision Loss": vision_loss_log[-1]}
    results_df = pd.concat(
        [pd.DataFrame([result_dictionary]), val_res], axis=1)
    return results_df
