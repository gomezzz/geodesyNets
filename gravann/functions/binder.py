from typing import Callable

import torch

_METHOD_REGISTRY = {
}


def bind_integration(method: str, use_acc: bool = True, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    pass


def bind_noise() -> Callable[[torch.Tensor], torch.Tensor]:
    pass
