import pathlib

from gravann.polyhedral import ACC_L as POLYHEDRAL_ACC_L
from . import load_polyhedral_mesh, load_mascon_data
from ._encodings import *
from ._mascon_labels import ACC_L as MASCON_ACC_L
from ._sample_observation_points import get_target_point_sampler
from ._train import init_network
from ._utils import fixRandomSeeds, EarlyStopping


def init_environment(parameters: {str, any}) -> str:
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
        {parameters['ground_truth']}/
        {parameters['sample']}/
        LR={parameters['learning_rate']}_loss={parameters['parameters'].__name__}_ENC={parameters['encoding'].name}_
        BS={parameters['batch_size']}_layers={parameters['hidden_layers']}_neurons={parameters['n_neurons']}_
        METHOD={parameters['sample_method']}_DOMAIN={domain}/
        """
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)
    return run_folder


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
        limit_shape_to_asteroid=f"f3dmeshes/{sample}_lp.pk"
    )


def init_prediction_label(integrator, model, encoding, integration_points, integration_domain):
    """Inits the prediction labels by binding relevant data to the evaluation function.

    Args:
        integrator: the integrator, e.g. trapezoid rule
        model: the neural network to bind
        encoding: the network's encoding
        integration_points: the number of integration points
        integration_domain: the intervall

    Returns:
        prediction function taking measurement points as input

    """
    return lambda points: integrator(points, model, encoding, N=integration_points, domain=integration_domain)


def init_ground_truth_labels(ground_truth, sample):
    """Inits the ground truth labels by binding relevant data from the sample to the evaluation function.

    Args:
        ground_truth: either 'polyhedral' or 'mascon'
        sample: the sample body's name

    Returns:
        label function taking measurement points as input

    """
    if ground_truth == 'polyhedral':
        mesh_vertices, mesh_edges = load_polyhedral_mesh(sample)
        return _init_polyhedral_label(mesh_vertices, mesh_edges)
    elif ground_truth == 'mascon':
        mascon_points, mascon_masses_u = load_mascon_data(sample)
        return _init_polyhedral_label(mascon_points, mascon_masses_u)
    else:
        raise NotImplemented(f"The method {ground_truth} is not implemented!")


def _init_polyhedral_label(mesh_vertices, mesh_edges):
    """Inits the polyhedral labels by binding mesh_vertices and mesh_edges to the evaluation function.
    """
    return lambda points: POLYHEDRAL_ACC_L(points, mesh_vertices, mesh_edges)


def _init_mascon_label(mascon_points, mascon_masses):
    """Inits the mascon labels by binding mascon_points and (uniform) mascon_masses to the evaluation function.
    """
    return lambda points: MASCON_ACC_L(points, mascon_points, mascon_masses)


def _init_noise(ground_truth_labels):
    """Inits noise by adding a noise function on top of the given ground truth labeling function.

    Args:
        ground_truth_labels: the polyhedral or mascon function only taking points as single argument

    Returns:
        None

    """
    pass
