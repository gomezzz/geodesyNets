import os
import pickle

import torch


def load_models(root_directory) -> list:
    models = []
    for dirpath, _, filenames in os.walk(root_directory):
        model, config = None, None
        for filename in filenames:
            if filename.endswith("best_model.mdl"):
                model = torch.load(os.path.join(dirpath, filename))
            elif filename.endswith("config.pk"):
                with open(os.path.join(dirpath, filename), 'rb') as file:
                    config = pickle.load(file)
        if model is not None:
            models.append((config, model))
    return models


m = load_models("../../results")

print(m)
