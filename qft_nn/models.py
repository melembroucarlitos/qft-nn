from typing import List, Dict, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from dataclasses import dataclass
import pathlib
from pydantic import BaseModel
from abc import ABC, abstractmethod
import einops

from qft_nn.config import ExperimentConfig

class SectionedCrossEntropy(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        # Reshape to [batch_size, num_sections, num_labels]
        pred = pred.view(-1, self.num_labels)
        target = target.view(-1, self.num_labels)
        
        pred_dist = F.softmax(pred, dim=1)
        target_dist = F.softmax(target, dim=1)
        
        # Compute cross entropy for each section
        losses = self.cross_entropy(pred_dist, target_dist)
        
        # Return average loss
        return losses.mean()

Criterion = Literal["cross_entropy", "sectioned_cross_entropy", "mse"]
CRITERION_DICT = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "sectioned_cross_entropy": SectionedCrossEntropy(num_labels=10),
    "mse": nn.MSELoss()
}
Optimizer = Literal["adam", "sgd"]
OPTIMIZER_DICT = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

class TrainConfig(ExperimentConfig):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    eval_every_n_batches: int = 100
    optimizer: Optimizer = "adam"
    criterion: Criterion = "cross_entropy"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Optional[pathlib.Path] = None

class MLPConfig(ExperimentConfig):
    input_dim: int = 784
    hidden_layers: List[int] = [64, 32]
    output_dim: int = 10

class EmbeddingBias(nn.Module):
    """CP from Logan's Implementation"""
    def __init__(self, embedding):
        super().__init__()
        num_tokens = embedding.weight.size(0)
        d_model = embedding.weight.size(1)
        self.bias = nn.Parameter(torch.zeros(num_tokens, d_model))
        self.bias.data.copy_(embedding.weight)
        self.bias.requires_grad = True

    def forward(self, x):
        return self.bias[x]

class Model(ABC, nn.Module):   
    def __init__(self, config: BaseModel):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def optimize(
        self,
        config: TrainConfig,
        train_loader: DataLoader,
        test_loader: DataLoader,
        eval_metric: str = "accuracy",
        save_dir: Optional[pathlib.Path] = None
    ) -> List[float]:
        if save_dir is not None and save_dir.exists():
            raise FileExistsError(f"Save directory already exists at {save_dir}")

        model = self.to(config.device)

        optimizer = OPTIMIZER_DICT[config.optimizer](self.parameters(), lr=config.learning_rate)
        criterion = CRITERION_DICT[config.criterion]

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=False)
            # Save train config
            config_path = save_dir / "train_config.json"
            with open(config_path, "w") as f:
                f.write(config.json())

        losses = []
        eval_metric_values = []
        for epoch in range(config.epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(config.device), target.to(config.device)
                
                model.train()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                if batch_idx % config.eval_every_n_batches == 0:
                    # Switch to eval mode for evaluation
                    model.eval()
                    eval_metric_value = self.evaluate(test_loader, config.device, eval_metric, criterion)
                    eval_metric_values.append(eval_metric_value)
                    print(f'Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Eval {eval_metric}: {eval_metric_value:.4f}')
                    
                    if save_dir is not None:
                        # Save model checkpoint
                        model_path = save_dir / f"model_epoch{epoch+1}_batch{batch_idx}.pt"
                        torch.save(self.state_dict(), model_path)
                    
                    # Switch back to train mode
                    model.train()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}')
            
            if save_dir is not None:
                # Save model after each epoch
                model_path = save_dir / f"model_epoch{epoch+1}.pt"
                torch.save(self.state_dict(), model_path)
                
        return losses, eval_metric_values
    
    def evaluate(self, test_loader: DataLoader, device: str, metric: str, criterion: Criterion) -> float:
        model = self.to(device)

        model.eval()
        output_value = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if metric == "accuracy":
                    pred = output.argmax(dim=1, keepdim=True)
                    output_value += pred.eq(target.view_as(pred)).sum().item()
                elif metric == "loss":
                    output_value += criterion(output, target).item()
                else:
                    raise ValueError(f"Invalid metric: {metric}")                    
        output_value /= len(test_loader.dataset) if metric == "accuracy" else len(test_loader)
        return output_value

    def flatten(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.state_dict().values()])

class MLP(Model):
    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_layers[0])
        self.fc2 = nn.Linear(config.hidden_layers[0], config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TopKDictionaryConfig(ExperimentConfig):
    input_dim: int
    latent_dim: int
    output_dim: int
    k: int
    activation: Literal["relu", "gelu", "id"] = "relu"
    encoder_bias: bool = True
    decoder_bias: bool = True
    input_center: Optional[torch.Tensor] = None
    output_center: Optional[torch.Tensor] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {
        "arbitrary_types_allowed": True
    }

class TopKDictionary(Model):
    """
    Adapted from Logan's Implementation, inspired by https://arxiv.org/abs/2406.04093

    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the TopKDictionary class docstring in that branch.

    With some additional effort, you can train a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    ACTIVATION_DICT = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "id": nn.Identity()
    }

    def __init__(self, config: TopKDictionaryConfig):
        super().__init__(config=config)
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.k = config.k
        self.activation = self.ACTIVATION_DICT[config.activation]

        self.encoder = nn.Linear(config.input_dim, config.latent_dim, bias=config.encoder_bias)
        self.decoder = nn.Linear(config.latent_dim, config.output_dim, bias=config.decoder_bias) # TODO: Do we have to be careful about initializaitons??

        self.set_decoder_norm_to_unit_norm() # TODO: Is this still important in the asymetrical case??

        if config.input_center is not None:
            self.input_center = config.input_center.to(self.config.device)
        else:
            self.input_center = torch.zeros(config.input_dim, device=self.config.device)

        if config.output_center is not None:
            self.output_center = config.output_center.to(self.config.device)
        else:
            self.output_center = torch.zeros(config.output_dim, device=self.config.device)

    def encode(self, x: torch.Tensor, return_topk: bool = False):
        x = x.to(self.config.device)
        input_centered = x - self.input_center
        post_activation = self.activation(self.encoder(input_centered))
        post_topk = post_activation.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_activation)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) - self.output_center

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    def evaluate(self, test_loader: DataLoader, device: str, metric: str, criterion: Criterion) -> float:
        model = self.to(device)
        model.eval()
        total_loss = 0
        total_fvu = 0
        num_batches = 0

        def calculate_fvu(x_orig, x_pred):
            """Calculate Fraction of Variance Unexplained"""
            mean = x_orig.mean(dim=0, keepdim=True)
            numerator = torch.mean(torch.sum((x_orig - x_pred)**2, dim=-1))
            denominator = torch.mean(torch.sum((x_orig - mean)**2, dim=-1))
            return numerator / (denominator + 1e-6)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.mse_loss(output, target)
                fvu = calculate_fvu(target, output)
                total_loss += loss.item()
                total_fvu += fvu.item()
                num_batches += 1

        print(f"Average MSE Loss: {total_loss/num_batches:.6f}")
        print(f"Average FVU: {total_fvu/num_batches:.6f}")
        return total_loss / num_batches

if __name__ == "__main__":
    def test_train_model(train_config: TrainConfig, model_config: BaseModel, model_type: Literal["mlp", "autoencoder"]):
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            transforms.Lambda(lambda x: x.flatten())  # Flatten the image
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False)

        # Create model based on type
        if model_type == "mlp":
            model = MLP(model_config)
        else:  # autoencoder
            model = TopKDictionary(config=model_config)

        # Train model
        losses, eval_accuracies = model.optimize(train_config, train_loader, test_loader)
        print("\nTraining completed!")
        print(f"Final evaluation accuracy: {eval_accuracies[-1]:.4f}")
        return losses, eval_accuracies

    # Prepare configurations
    train_config = TrainConfig(
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="cross_entropy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    mlp_config = MLPConfig(
        input_dim=784,  # 28x28 MNIST images
        hidden_layers=[128, 64],  # Two hidden layers
        output_dim=10  # 10 classes for MNIST
    )

    topk_dictionary_config = TopKDictionaryConfig(
        input_dim=784,  # 28x28 MNIST images flattened
        latent_dim=128,
        output_dim=784,  # Same as input for reconstruction
        k=10,
        activation="relu",
        encoder_bias=True,
        decoder_bias=True,
        input_center=None,
        output_center=None
    )

    # # Train MLP
    print("Training MLP...")
    mlp_losses, mlp_accuracies = test_train_model(train_config, mlp_config, "mlp")

    # Train AutoEncoder
    # print("\nTraining AutoEncoder...")
    # ae_losses, ae_accuracies = test_train_model(train_config, topk_dictionary_config, "autoencoder")

