import torch
import numpy as np
from jaxtyping import Float
from typing import Callable


def compute_empirical_ntk(
    model: torch.nn.Module,
    loss: Callable, # TODO: Flesh out full type signature
    data: Float[torch.Tensor, "data_dim num_data"],
) -> Float[np.ndarray, "num_data num_data"]:
    raise NotImplementedError
