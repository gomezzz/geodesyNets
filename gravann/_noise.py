from typing import Callable

import torch.distributions
from torch import Tensor


def get_noise(name: str, label_fn: Callable[[Tensor], Tensor], **kwargs) -> Callable[[Tensor], Tensor]:
    """Adds noise to a labeling function's output.

    Args:
        name: the name of the noise method, either 'gaussian' or 'constant_bias'
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        **kwargs: additional parameters (depends on chosen noise)

    Keyword Args:
        mean (float): only if 'gaussian', defaults to 0.0
        std (float): only if 'gaussian', defaults to 1.0
        bias (Tensor): only if 'constant_bias', defaults to Tensor([0, 1, 0])

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    if name == 'gaussian':
        return _gaussian_noise(label_fn, **{k: v for k, v in kwargs.items() if k in ['mean', 'std']})
    elif name == 'constant_bias':
        return _constant_bias(label_fn, **{k: v for k, v in kwargs.items() if k in ['bias']})
    else:
        raise NotImplementedError(f"The noise method {name} is not implemented!")


def _gaussian_noise(
        label_fn: Callable[[Tensor], Tensor],
        mean: float = 0.0,
        std: float = 1.0
) -> Callable[[Tensor], Tensor]:
    """Adds Gaussian noise to a label function.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        mean: mean value of the gaussian
        std: std value of the gaussian

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    return lambda points_tensor: \
        label_fn(points_tensor) + torch.distributions.Normal(mean, std).sample(points_tensor.shape)


def _constant_bias(
        label_fn: Callable[[Tensor], Tensor],
        bias: Tensor = torch.Tensor([0, 1, 0])
) -> Callable[[Tensor], Tensor]:
    """Adds a constant bias noise to a label function.

    Args:
        label_fn: function taking a tensor and producing the labels, the base function on which the noise is added
        bias: Adds a constant bias on each result of the label function, e.g. Tensor[0, 1, 0]

    Returns:
        function taking an input tensor and evaluating labels + adding noise on the results

    """
    return lambda points_tensor: \
        label_fn(points_tensor) + bias
