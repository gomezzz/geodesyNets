import pathlib

# Required for loading runs
from ._encodings import *
from ._losses import contrastive_loss, normalized_relative_L2_loss, normalized_relative_component_loss
from ._sample_observation_points import get_target_point_sampler
from ._train import init_network
from ._utils import fixRandomSeeds, EarlyStopping


def _init_environment(parameters: {str, any}):
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Fix the random seeds for this run
    fixRandomSeeds()

    domain = str(parameters['sample_domain']) \
        .replace('.', '_').replace('[', '').replace(']', '').replace(',', '').replace(' ', '=')

    # Create folder for this specific run
    run_folder = f"""
        {parameters['output_folder']}/
        {parameters['sample']}/
        LR={parameters['learning_rate']}_loss={parameters['parameters'].__name__}_ENC={parameters['encoding'].name}_
        BS={parameters['batch_size']}_layers={parameters['hidden_layers']}_neurons={parameters['n_neurons']}_
        METHOD={parameters['target_sample_method']}_DOMAIN={domain}/
        """
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)
    return run_folder


def _init_model(run_folder: str, encoding: any, n_neurons: int, activation: any, model_type: str, omega: float,
                hidden_layers: int, learning_rate: float):
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
    return get_target_point_sampler(
        batch_size,
        method=target_sample_method,
        bounds=sample_domain,
        limit_shape_to_asteroid=f"f3dmeshes/{sample}_lp.pk"
    )


def _train_on_batch_v2(targets, labels, model, encoding, loss_fn, optimizer, scheduler, integrator, N,
                       integration_domain=None):
    """Trains the passed model on the passed batch

    Args:
        targets (tensor): target points for training
        labels (tensor): labels at the target points
        model (torch model): model to train
        encoding (func): encoding function for the model
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use
        integrator (func): integration function to call for the training loss
        N (int): Number of integration points to use for training
        integration_domain (torch.tensor): Domain to pick integration points in, only works with trapezoid for now

    Returns:
        torch tensor: losses
    """
    # Compute the loss (use N=3000 to start with, then, eventually, beef it up to 200000)
    predicted = integrator(targets, model, encoding, N=N, domain=integration_domain)
    c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)
    if loss_fn == contrastive_loss or loss_fn == normalized_relative_L2_loss or loss_fn == normalized_relative_component_loss:
        loss = loss_fn(predicted, labels)
    else:
        loss = loss_fn(predicted.view(-1), labels.view(-1))

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Perform a step in LR scheduler to update LR
    scheduler.step(loss.item())

    return loss, c


def run_training_v2():
    pass
