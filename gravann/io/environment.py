import pathlib

import numpy as np
import torch


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
