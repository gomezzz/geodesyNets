import os
import pickle

import torch

from gravann import init_network, direct_encoding
from gravann.network._abs_layer import AbsLayer


def read_data(root_directory) -> list:
    models = []
    for dirpath, _, filenames in os.walk(root_directory):
        model_stats, config = None, None
        for filename in filenames:
            if filename.endswith("best_model.mdl"):
                model_stats = torch.load(os.path.join(dirpath, filename))
            elif filename.endswith("config.pk"):
                with open(os.path.join(dirpath, filename), 'rb') as file:
                    config = pickle.load(file)
        if model_stats is not None:
            models.append((config, model_stats))
    return models


def populate_models(models: list) -> list:
    models = []
    for config, model_stats in models:
        models.append((config, populate_model(config, model_stats)))
    return models


def populate_model(cfg: dict, model_stats: dict) -> torch.nn.Module:
    ENCODING_TMP_REGISTRY = {
        "direct_encoding": direct_encoding
    }
    encoding = ENCODING_TMP_REGISTRY[cfg["Encoding"]]()

    if cfg["Activation"] == "AbsLayer":
        activation = AbsLayer()
    else:
        activation = getattr(torch.nn, cfg["Activation"])()

    # TODO Adapt omega if it is different!
    model = init_network(encoding, model_type=cfg["Model"], activation=activation, n_neurons=cfg["n_neurons"], hidden_layers=cfg["hidden_layers"], siren_omega=30.0)
    model.load_state_dict(model_stats)

    return model



m = read_data("F:\\results")


c = populate_model(m[0][0], m[0][1])

print(c)
