from collections.abc import Callable

import torch


def dict_apply(
    x: dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Recursively apply a function to all tensors in a nested dictionary.
    """
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
