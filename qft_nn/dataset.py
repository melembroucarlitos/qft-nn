from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
from typing import List, Tuple, Optional
import numpy as np
import torch.nn as nn
import time
from torchvision import datasets, transforms

from qft_nn.sgld import _sgld_parallel, SGLDConfig
from qft_nn.models import MLP, MLPConfig, TrainConfig

def _create_mnist_dataloaders(batch_size: int, shuffle: bool) -> Tuple[DataLoader, DataLoader]:
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        transforms.Lambda(lambda x: x.flatten())  # Flatten the image
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    eval_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, eval_dataloader


class DictionaryDataset(Dataset):
    def __init__(self, data: List[tuple]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def load_from_file(cls, file_path: pathlib.Path) -> "DictionaryDataset":
        return cls(torch.load(file_path))

    def save_to_file(self, file_path: pathlib.Path):
        torch.save(self.data, file_path)

def _create_vectorized_model_function(model: nn.Module, dataloader: DataLoader, device: str) -> torch.Tensor:
    out = []
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            out.append(logits.detach().flatten())  # Detach the tensor
    return torch.cat(out)

def _create_mlp_to_vectorized_model_function_dataset(mlp: nn.Module, dataloader: DataLoader, sgld_config: SGLDConfig, n_models: int, train_eval_split: float, device: str, dir_path: Optional[pathlib.Path] = None) -> Tuple[DictionaryDataset, DictionaryDataset, DataLoader, DataLoader]:
    # Check CUDA availability and set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    # Create n_models copies of the MLP
    models = []
    for _ in range(n_models):
        model = MLP(mlp.config)
        model = model.to(device)
        model.load_state_dict(mlp.state_dict())
        models.append(model)

    # Run parallel SGLD
    models = _sgld_parallel(models, dataloader, device, sgld_config)
    print(f"Total SGLD sampling took {time.time() - start_time:.2f} seconds")
    
    # Process models in batches on GPU
    with torch.no_grad():
        # First get flattened parameters for all models
        flattened_models = torch.stack([model.flatten() for model in models])
        
        # Process full dataset through all models in smaller batches
        vectorized_outputs = []
        batch_size = n_models  # Process 8 models at a time to manage memory
        
        for model_batch_idx in range(0, len(models), batch_size):
            # Get current batch of models
            model_batch = models[model_batch_idx:model_batch_idx + batch_size]
            batch_outputs = []
            
            # Run each batch of models through the full dataset
            for data, _ in dataloader:
                data = data.to(device)
                # Forward pass through batch of models simultaneously
                outputs = torch.stack([model(data).detach().flatten() for model in model_batch])
                batch_outputs.append(outputs)
                
                # Clear cache after each batch to manage memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all batches for this group of models
            model_outputs = torch.cat(batch_outputs, dim=1)  # Concatenate along feature dimension
            vectorized_outputs.append(model_outputs)
            
            # Clear cache after processing each model batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Stack outputs from all model batches
        vectorized_models = torch.cat(vectorized_outputs, dim=0)
        
        # Create training data pairs
        autoencoder_training_data = list(zip(flattened_models, vectorized_models))

    train_size = int(train_eval_split * len(autoencoder_training_data))
    eval_size = len(autoencoder_training_data) - train_size
    train_data, eval_data = torch.utils.data.random_split(
        autoencoder_training_data, 
        [train_size, eval_size]
    )

    if dir_path is not None:
        torch.save(train_data, dir_path / "autoencoder_train_dataset.pt")
        torch.save(eval_data, dir_path / "autoencoder_eval_dataset.pt")

    return DictionaryDataset(train_data), DictionaryDataset(eval_data)

def _create_and_save_dataset(mlp_config: MLPConfig, mlp_train_config: TrainConfig, sgld_config: SGLDConfig, n_models: int, train_eval_split: float, device: str, dir_path: pathlib.Path) -> Tuple[DictionaryDataset, DictionaryDataset]:
    mlp = MLP(mlp_config)
    mnist_train_dataloader, mnist_eval_dataloader = _create_mnist_dataloaders(10, True)
    mlp.optimize(mlp_train_config, mnist_train_dataloader, mnist_eval_dataloader)
    dictionary_train_dataset, dictionary_eval_dataset = _create_mlp_to_vectorized_model_function_dataset(mlp,
                                                                                                         mnist_train_dataloader, 
                                                                                                         sgld_config, 
                                                                                                         10, 
                                                                                                         0.8, 
                                                                                                         mlp_train_config.device, 
                                                                                                         pathlib.Path(pathlib.Path(dir_path)))

if __name__ == "__main__":
    
    mlp_config = MLPConfig(
        input_dim=784,
        hidden_layers=[128, 64],
        output_dim=10
    )
    
    mlp_train_config = TrainConfig(
        epochs=1,
        batch_size=128,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="cross_entropy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    sgld_config = SGLDConfig(
        learning_rate=0.001, 
        batch_size=10, 
        criterion="cross_entropy", 
        optimizer="sgd", 
        temperature=0.01, 
        num_steps=2
    )
