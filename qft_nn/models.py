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

class TrainConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    eval_every_n_batches: int = 100
    optimizer: Optimizer = "adam"
    criterion: Criterion = "cross_entropy"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Optional[pathlib.Path] = None

class MLPConfig(BaseModel):
    input_dim: int = 784
    hidden_layers: List[int] = [64, 32]
    output_dim: int = 10

class AutoencoderConfig(BaseModel):
    activation_dim: int
    dict_size: int
    k: int
    data_mean: Optional[torch.Tensor] = None
    tokens_to_combine: Optional[torch.Tensor] = None
    embedding: Optional[nn.Module] = None

class EmbeddingBias(nn.Module):
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
        x = x.view(x.size(0), -1)  # Flattens the input TODO: Move this into the mnist dataset/dataloader creation
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AutoEncoderTopK(nn.Module):
    """
    (Adapted from Logan's Implementation, notes below are cp)
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can traisn a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.k = config.k
        self.tokens_to_combine = config.tokens_to_combine

        self.encoder = nn.Linear(config.activation_dim, config.dict_size)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(config.dict_size, config.activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(config.activation_dim))
        self.per_token_bias = config.embedding

    def encode(self, x: torch.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

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

    @classmethod
    def from_pretrained(cls, path: str, k: int, device: Optional[str] = None, embedding: Optional[nn.Module] = None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path, weights_only=True, map_location='cpu')
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        
        if 'per_token_bias.bias' in state_dict:
            num_tokens, d_model = state_dict["per_token_bias.bias"].shape
            embedding = EmbeddingBias(nn.Embedding(num_tokens, d_model))
        else: 
            embedding = None
            
        config = AutoencoderConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            embedding=embedding
        )
        
        autoencoder = cls(config)
        autoencoder.load_state_dict(state_dict)
        
        if device is not None:
            autoencoder.to(device)
            
        return autoencoder
    
if __name__ == "__main__":
    # Set up configurations
    mlp_config = MLPConfig(
        input_dim=784,  # 28x28 MNIST images
        hidden_layers=[128, 64],  # Two hidden layers
        output_dim=10  # 10 classes for MNIST
    )
    
    train_config = TrainConfig(
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="cross_entropy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False)

    # Create and train MLP
    model = MLP(mlp_config.input_dim, mlp_config.hidden_layers[0], mlp_config.output_dim)
    losses, eval_accuracies = model.optimize(train_config, train_loader, test_loader)
    
    print("\nTraining completed!")
    print(f"Final evaluation accuracy: {eval_accuracies[-1]:.4f}")
