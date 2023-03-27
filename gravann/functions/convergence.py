import numpy as np

from gravann.input import model_reader
from gravann.labels import binder as label_binder
from gravann.util import get_target_point_sampler
from . import binder as function_binder


def convergence_test(model_root_path, n_list, max_bound=1.0):
    model, cfg = model_reader.read_models(model_root_path)
    label_fn = label_binder.bind_label('polyhedral', cfg["use_acc"], sample=cfg["sample"])
    target_points_sampler = get_target_point_sampler(
        100,
        method='spherical',
        bounds=[0.0, max_bound],
        limit_shape_to_asteroid=f"./3dmeshes/{cfg['sample']}_lp.pk"
    )
    target_points = target_points_sampler()
    errors = []
    for n in n_list:
        integration_fn = function_binder.bind_integration(
            method='trapezoid',
            use_acc=True,
            model=model,
            encoding=cfg["encoding"],
            integration_points=n,
            integration_domain=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        )
        errors.append(_convergence_test_single(label_fn, integration_fn, target_points))

    for i in range(1, len(errors)):
        p = np.log(errors[i] / errors[i - 1]) / np.log(n_list[i] / n_list[i - 1])
        print(f"p={p:.2f} for n={n_list[i]}")


def _convergence_test_single(label_fn, integration_fn, target_points):
    expected_values = label_fn(target_points)
    actual_values = integration_fn(target_points)

    return abs(actual_values - expected_values).mean()
