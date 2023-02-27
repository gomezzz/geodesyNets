import inspect
from typing import Callable, Union

import torch

import gravann.network.encodings
from . import integration, noise

_INTEGRATION_METHOD_REGISTRY = {
    ("trapezoid", True): integration.acceleration_trapezoid,
    ("trapezoid", False): integration.potential_trapezoid
}


def bind_integration(
        method: str, use_acc: bool, model,
        encoding: gravann.network.encodings, **kwargs
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Binds arguments to a given integration function, so that it can be called by just inputting the tensor of points.

    Args:
        method: the integration method, e.g. 'trapezoid'
        use_acc: if true, use the acceleration else use the potential
        model: the neural model to bind the integration
        encoding: the utilized encoding of the network
        **kwargs: arguments to bind

    Keyword Args:
        integration_points (int): number of integration points (required for trapezoid)
        integration_domain (torch.tensor): integration domain [3,2]

    Returns:
        callable function taking an input tensor of points and returning the corresponding label's values

    """
    if method == 'trapezoid':
        return lambda points: _INTEGRATION_METHOD_REGISTRY[(method, use_acc)](
            points,
            model,
            encoding,
            N=kwargs.get("integration_points", 50000),
            domain=kwargs.get('integration_domain', None)
        )
    else:
        raise NotImplementedError(f"The method {method} is not implemented")


_NOISE_METHOD_REGISTRY = {
    "gaussian": noise.gaussian_noise,
    "constant_bias": noise.constant_bias
}


def bind_noise(
        method: Union[str, None], label_fn: Callable[[torch.Tensor], torch.Tensor], **kwargs
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Binds arguments (adding noise) to a function taking input points.

    Args:
        method: the noise to add
        label_fn: the function to concat a noise function evaluation
        **kwargs: parameters of the chosen noise function

    Keyword Args:
        mean (float): only if 'gaussian', defaults to 0.0
        std (float): only if 'gaussian', defaults to 1.0
        bias (Tensor): only if 'constant_bias', defaults to Tensor([0, 1, 0])


    Returns:
        callable function taking an input tensor of points and returning the corresponding label's values
    """
    if method is None or method == "":
        return label_fn
    try:
        noise_method = _NOISE_METHOD_REGISTRY[method]
        return noise_method(
            label_fn, **{k: v for k, v in kwargs.items() if k in inspect.getfullargspec(noise_method).args}
        )
    except KeyError:
        raise NotImplementedError(f"The chosen {method} does not exist/ is not correctly implemented!")
