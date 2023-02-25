import pathlib
from typing import Callable, Union

import numpy as np
import torch

from gravann.functions import integration, noise
from gravann.input import sample_reader
from gravann.labels import mascon, polyhedral, calculate_density
from gravann.network import network_initalizer
from gravann.util import EarlyStopping, get_target_point_sampler


def get_run_folder(cfg: dict, create_folder: bool = False) -> str:
    """Returns a run folder path as string for a given configuration.
    Args:
        cfg: dictionary with the configuration
        create_folder: if the run_folder should also be created for the system

    Returns:
        path as string

    """
    domain = str(cfg['sample_domain']) \
        .replace('.', '_').replace('[', '').replace(']', '').replace(',', '').replace(' ', '=')

    run_folder = (
        f"{cfg['output_folder']}/"
        f"{cfg['sample']}/"
        f"{cfg['ground_truth']}/"
        f"SEED={cfg['seed']}_LR={cfg['learning_rate']}_LOSS={cfg['loss_fn'].__name__}_"
        f"ENC={cfg['encoding'].name}_ACT={str(cfg['activation'])[:-2]}_"
        f"BS={cfg['batch_size']}_LAYERS={cfg['hidden_layers']}_NEURONS={cfg['n_neurons']}_"
        f"METHOD={cfg['sample_method']}_DOMAIN={domain}_NOISE={cfg['noise_method']}/"
    )

    if create_folder:
        pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)

    return run_folder


def init_cuda_environment(cfg: dict) -> None:
    """Empties the cuda cache and inits random seed generators of numpy and torch.

    Args:
        cfg: parameters, should contain an entry {'seed': _}

    Returns:
        None

    """
    torch.cuda.empty_cache()
    # Fix the random seeds for this run
    _fixate_seeds(cfg['seed'])


def _fixate_seeds(seed: int = 42):
    """This function sets the random seeds in torch and numpy to enable reproducible behavior.

    Args:
        seed (optional): the chosen seed, defaults to 42
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def init_environment(parameters: dict) -> str:
    """Creates the environment (the run-folder) for the given training with parameters

    Args:
        parameters: dictionary of parameters of this training run

    Returns:
        the run folder

    """
    init_cuda_environment(parameters)
    return get_run_folder(parameters, create_folder=True)


def init_model_and_optimizer(run_folder: str, encoding: any, n_neurons: int, activation: any, model_type: str,
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
    model = network_initalizer.init_network(
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


def init_training_sampler(sample: str, target_sample_method: str, sample_domain: [float], batch_size: int):
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
        limit_shape_to_asteroid=f"3dmeshes/{sample}_lp.pk"
    )


def init_prediction_label(model, encoding, integration_points, integration_domain, use_acc=True):
    """Inits the prediction labels by binding relevant data to the evaluation function.

    Args:
        model: the neural network to bind
        encoding: the network's encoding
        integration_points: the number of integration points
        integration_domain: the intervall
        use_acc: calculate the acceleration rather than potential

    Returns:
        prediction function taking measurement points as input

    """
    if use_acc:
        return lambda points: integration.acceleration_trapezoid(points, model, encoding, N=integration_points,
                                                                 domain=integration_domain)
    else:
        return lambda points: integration.potential_trapezoid(points, model, encoding, N=integration_points)


def init_input_data(sample: str) -> dict:
    """Reads the input mesh and mascon data for a given sample.

    Args:
        sample: the body's name

    Returns:
        dictionary conatining the input data

    """
    mesh_vertices, mesh_faces = sample_reader.load_polyhedral_mesh(sample)
    mascon_points, mascon_masses_u = sample_reader.load_mascon_data(sample)
    return {
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "mascon_points": mascon_points,
        "mascon_masses": mascon_masses_u
    }


def init_ground_truth_labels(ground_truth: str, input_data: dict, use_acc=True):
    """Inits the ground truth labels by binding relevant data from the sample to the evaluation function.

    Args:
        ground_truth: either 'polyhedral' or 'mascon'
        input_data: dictionary containing mesh and mascon information
        use_acc: use ACC_L instead of U_L

    Returns:
        label function taking measurement points as input and input read

    """
    if ground_truth == 'polyhedral':
        density = calculate_density(input_data["mesh_vertices"], input_data["mesh_faces"])
        return _init_polyhedral_label(input_data["mesh_vertices"], input_data["mesh_faces"], density, use_acc)
    elif ground_truth == 'mascon':
        return _init_mascon_label(input_data["mascon_points"], input_data["mascon_masses"], use_acc)
    else:
        raise NotImplemented(f"The method {ground_truth} is not implemented!")


def _init_polyhedral_label(mesh_vertices, mesh_edges, density, use_acc=True):
    """Inits the polyhedral labels by binding mesh_vertices and mesh_edges to the evaluation function.
    """
    if use_acc:
        return lambda points: polyhedral.acceleration(points, mesh_vertices, mesh_edges, density)
    else:
        return lambda points: polyhedral.potential(points, mesh_vertices, mesh_edges, density)


def _init_mascon_label(mascon_points, mascon_masses, use_acc=True):
    """Inits the mascon labels by binding mascon_points and (uniform) mascon_masses to the evaluation function.
    """
    if use_acc:
        return lambda points: mascon.acceleration(points, mascon_points, mascon_masses)
    else:
        return lambda points: mascon.potential(points, mascon_points, mascon_masses)


def init_noise(
        label_fn: Callable[[torch.Tensor], torch.Tensor], method: Union[str, None], method_param: dict
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Inits noise by adding a noise function on top of the given ground truth labeling function.

    Args:
        label_fn: the labeling function taking points producing the ground truth
        method: the chosen method to generate noise
        method_param: the additional parameters of the chosen method


    Returns:
        The label function with noise addition

    """
    if method is None or method == "":
        return label_fn
    else:
        return noise.add(method, label_fn, **method_param)
