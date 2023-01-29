import pathlib

# Required for loading runs
from ._encodings import *
from ._losses import contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss
from ._sample_observation_points import get_target_point_sampler
from ._train import init_network
from ._utils import fixRandomSeeds, EarlyStopping


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


def _init_model(run_folder: str, encoding: any, n_neurons: int, activation: any, model_type: str, omega: float,
                hidden_layers: int, learning_rate: float):
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

    # Initializes training utility
    early_stopper = EarlyStopping(
        save_folder=run_folder
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
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
        sample_domain: the sample domain, sspecifies the sampling radius.
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


def _train_on_batch_v2(points, model_fn, labels, loss_fn, optimizer, scheduler):
    """Trains the passed model on the passed batch

    Args:
        points (tensor): target points for training
        model_fn (func): prediction func of the model
        labels (tensor): labels at the target points
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use

    Returns:
        torch tensor: losses
    """
    # Compute the loss (use N=3000 to start with, then, eventually, beef it up to 200000)
    # predicted = integrator(points, model, encoding, N=N, domain=integration_domain)
    predicted = model_fn(points)
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


def run_training_v2():
    pass
