# Adapted from ARENA's TMS & SAE Solution notebook, (https://colab.research.google.com/drive/1rPy82rL3iZzy2_Rd3F82RwFhlVnnroIh?usp=sharing, March 11 2025)

from typing import Callable, Optional
from jaxtyping import Float
import torch
import torch.nn.functional as F
from tqdm import tqdm
import einops

from qft_nn.nn.base_config import Config, TrainConfig
from qft_nn.nn.lr_schedules import constant_lr
from qft_nn.nn.toy_model import SingleLayerToyReLUModel, SingleLayerToyReLUModelConfig

class AutoEncoderConfig(Config):
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 1.0
    tied_weights: bool = False
    train: TrainConfig

class AutoEncoder(torch.nn.Module):
    W_enc: Float[torch.Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[torch.Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[torch.Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[torch.Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig, device: torch.types.Device):
        super().__init__()
        self.cfg = cfg
        self.W_enc = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.n_instances, cfg.n_hidden_ae))
        if not(cfg.tied_weights):
            self.W_dec = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.n_instances, cfg.n_input_ae))
        self.to(device)


    def forward(self, h: Float[torch.Tensor, "batch_size n_instances n_hidden"]):
        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).sum(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)


    @torch.no_grad()
    def resample_neurons(
        self,
        h: Float[torch.Tensor, "batch_size n_instances n_hidden"],
        frac_active_in_window: Float[torch.Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        _, l2_loss, _, _, _ = self.forward(h)

        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_neurons_mask = torch.empty((self.cfg.n_instances, self.cfg.n_hidden_ae), dtype=torch.bool, device=self.W_enc.device)

        for instance in range(self.cfg.n_instances):

            # Find the dead neurons in this instance. If all neurons are alive, continue
            is_dead = (frac_active_in_window[:, instance].sum(0) < 1e-8)
            dead_neurons_mask[instance] = is_dead
            dead_neurons = torch.nonzero(is_dead).squeeze(-1)
            alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
            n_dead = dead_neurons.numel()
            if n_dead == 0: continue

            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance] # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue # If we have zero reconstruction loss, we don't need to resample neurons

            # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = torch.distributions.Categorical(probs = l2_loss_instance / l2_loss_instance.sum())
            replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance] # shape [n_dead n_input_ae]

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[instance, :, alive_neurons].norm(dim=0).mean().item()

            # Use this to renormalize the replacement values
            replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8)) * W_enc_norm_alive_mean * neuron_resample_scale

            # Lastly, set the new weights & biases
            self.W_enc.data[instance, :, dead_neurons] = replacement_values.T
            self.b_enc.data[instance, dead_neurons] = 0.0

        # Return data for visualising the resampling process
        colors = [["red" if dead else "black" for dead in dead_neuron_mask_inst] for dead_neuron_mask_inst in dead_neurons_mask]
        title = f"resampling {dead_neurons_mask.sum()}/{dead_neurons_mask.numel()} neurons (shown in red)"
        return colors, title


    def optimize(
        self,
        model: SingleLayerToyReLUModel,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        '''
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        '''
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"values": [], "colors": [], "titles": [], "frac_active": []}
        colors = None
        title = "no resampling yet"

        for step in progress_bar:

            # Normalize the decoder weights before each optimization step
            self.normalize_decoder()

            # Resample dead neurons
            if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = torch.stack(frac_active_list[-neuron_resample_window:], dim=0)
                # Compute batch of hidden activations which we'll use in resampling
                batch = model.generate_batch(batch_size)
                h = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")
                # Resample
                colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Get a batch of hidden activations from the model
            with torch.inference_mode():
                features = model.generate_batch(batch_size)
                h = einops.einsum(features, model.W, "... instances features, instances hidden features -> ... instances hidden")

            # Optimize
            optimizer.zero_grad()
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()

            # Calculate the sparsities, and add it to a list
            frac_active = einops.reduce((acts.abs() > 1e-8).float(), "batch_size instances hidden_ae -> instances hidden_ae", "mean")
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(l1_loss=l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
                data_log["values"].append(self.W_enc.detach().cpu())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu())

        return data_log
    
if __name__ == "__main__":
    train_config = TrainConfig(
        n_instances=1,
        batch_size=1024,
        steps=10_000,
        log_freq=100,
        lr=1e-3,
        lr_scale=constant_lr,
        data_seed=1337
    )

    toy_model_config = SingleLayerToyReLUModelConfig(
        n_features=5,
        n_hidden=32,
        n_correlated_pairs=2,
        n_anticorrelated_pairs=1,
        train=train_config
    )

    sae_config = AutoEncoderConfig(
        n_input_ae = 5,
        n_hidden_ae = 12,
        l1_coeff = 1.0,
        tied_weights = False,
        train=train_config,
    )

    toy_model = SingleLayerToyReLUModel(cfg=toy_model_config, device='cpu')
    sae = AutoEncoder(cfg=sae_config, device='cpu')
    sae.optimize(model=toy_model)
