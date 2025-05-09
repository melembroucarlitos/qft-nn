from typing import List, Dict, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
from abc import ABC, abstractmethod

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

class MLPConfig(BaseModel):
    input_dim: int = 784
    hidden_layers: List[int] = [64, 32]
    output_dim: int = 10

class AutoencoderConfig(BaseModel):
    input_dim: int = 784
    encoder_layers: List[Dict[str, int | Literal["relu", "id"]]] = [{"dim": 64, "activation": "relu"}, {"dim": 32, "activation": "relu"}]
    decoder_layers: List[Dict[str, int | Literal["relu", "id"]]] = [{"dim": 64, "activation": "relu"}, {"dim": 784, "activation": "id"}]


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
        eval_metric: str = "accuracy"
    ) -> List[float]:
        model = self.to(config.device)

        optimizer = OPTIMIZER_DICT[config.optimizer](self.parameters(), lr=config.learning_rate)
        criterion = CRITERION_DICT[config.criterion]

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
                    # Switch back to train mode
                    model.train()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}')
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


class Autoencoder(Model):
    def __init__(
        self,
        config: AutoencoderConfig
    ):
        super().__init__(config)
        
        # Build encoder
        encoder_modules = []
        prev_dim = config.input_dim
        for layer in config.encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, layer["dim"]))
            if layer["activation"] == "relu":
                encoder_modules.append(nn.ReLU())
            prev_dim = layer["dim"]
        
        # Build decoder
        decoder_modules = []
        for layer in config.decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, layer["dim"]))
            if layer["activation"] == "relu":
                decoder_modules.append(nn.ReLU())
            prev_dim = layer["dim"]
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

    

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
