import torch
import numpy as np

from gravann.input import model_reader
from gravann.labels import binder as label_binder
from gravann.util import get_target_point_sampler, deep_get
from . import binder as function_binder
from ..network import encodings


def error_calculation(model_root_path, n_list, max_bound=1.0):
    model, cfg = model_reader.read_models(model_root_path)[0]
    encoding = encodings.get_encoding(deep_get(cfg, ["Encoding", "encoding"]))
    sample = deep_get(cfg, ["Sample", "sample"])
    label_fn = label_binder.bind_label('polyhedral', True, sample=sample)
    target_points_sampler = get_target_point_sampler(
        100,
        method='spherical',
        bounds=[0.0, max_bound],
        limit_shape_to_asteroid=f"./3dmeshes/{sample}_lp.pk"
    )
    target_points = target_points_sampler()
    errors = []
    for n in n_list:
        integration_fn = function_binder.bind_integration(
            method='trapezoid',
            use_acc=True,
            model=model,
            encoding=encoding,
            integration_points=n,
            integration_domain=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        )
        errors.append(_calculate_relative_error(label_fn, integration_fn, target_points, c=cfg["c"].to("cpu")))
    print(errors)


def _calculate_relative_error(label_fn, integration_fn, target_points, c=1.0):
    true_value = label_fn(target_points)
    approximated_value = integration_fn(target_points) * c

    true_error = abs(true_value - approximated_value)

    return float(abs(true_error / true_value).mean())
