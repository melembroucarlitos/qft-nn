# Adapted from ARENA's TMS & SAE Solution notebook, (https://colab.research.google.com/drive/1rPy82rL3iZzy2_Rd3F82RwFhlVnnroIh?usp=sharing, March 11 2025)

from dataclasses import dataclass
from typing import Callable, Optional, Union
from einops import einops
import numpy as np
from jaxtyping import Float
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml


def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class SingleLayerToyReLUModelConfig:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int
    n_hidden: int
    n_correlated_pairs: int
    n_anticorrelated_pairs: int

    @classmethod
    def from_yaml(cls, filepath: str) -> "SingleLayerToyReLUModelConfig":
        with open(filepath, 'r') as file:
            config_dict = yaml.safe_load(file)

        if "SingleLayerToyReLUModelConfig" in config_dict:
            config_dict = config_dict["SingleLayerToyReLUModelConfig"]
            
        return cls(**config_dict)


class SingleLayerToyReLUModel(torch.nn.Module):
    W: Float[torch.Tensor, "n_instances n_hidden n_features"]
    b_final: Float[torch.Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: SingleLayerToyReLUModelConfig,
        device: torch.types.Device,
        feature_probability: Optional[Union[float, torch.Tensor]] = None,
        importance: Optional[Union[float, torch.Tensor]] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = torch.ones(())
        if isinstance(feature_probability, float): feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_instances, cfg.n_features))
        if importance is None: importance = torch.ones(())
        if isinstance(importance, float): importance = torch.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_instances, cfg.n_features))

        self.W = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = torch.nn.Parameter(torch.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)


    def forward(
        self,
        features: Float[torch.Tensor, "... instances features"]
    ) -> Float[torch.Tensor, "... instances features"]:
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)


    def generate_batch(self, batch_size) -> Float[torch.Tensor, "batch_size instances features"]:
        '''
        Generates a batch of data. We'll return to this function later when we apply correlations.
        '''
        feat = torch.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_seeds = torch.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = torch.where(
            feat_is_present,
            feat,
            torch.zeros((), device=self.W.device),
        )
        return batch


    def calculate_loss(
        self,
        out: Float[torch.Tensor, "batch instances features"],
        batch: Float[torch.Tensor, "batch instances features"],
    ) -> Float[torch.Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        '''
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        '''
        Optimizes the model using the given hyperparameters.
        '''
        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)