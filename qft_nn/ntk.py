import torch
import numpy
from jaxtyping import Float
from typing import Callable


def compute_empirical_ntk(
    model: torch.Model,
    loss: Callable[[], float],
    data: Float[torch.Tensor, "data_dim num_data"],
) -> Float[numpy.float, "num_data num_param"]:
    raise NotImplementedError
