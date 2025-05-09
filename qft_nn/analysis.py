from typing import List, Literal, Tuple, Optional
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass
import time
from qft_nn.models import MLP, Autoencoder, MLPConfig, AutoencoderConfig, TrainConfig, OPTIMIZER_DICT, CRITERION_DICT, Optimizer, Criterion
import torch.multiprocessing as mp

# TODO:
# - Model & Batch Parallelize the make_autoencoder_dataset function
# - Batch Parallelize the SGLD sampling by creating an sgld optimizer (??)
# - POC on AWS persistence

# - Fix _evaluate_bug
# - Test MLP reconstruction in create_autoencoder_dataset
# - Start up a dataset creation run on runpod
# - Run autoencoder sweeps

# - Robustify the model parallelizations with proper batching

class SGLDConfig(BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 64
    criterion: Criterion = "cross_entropy"
    optimizer: Optimizer = "sgd"
    temperature: float = 1
    num_steps: int = 10

def _sgld(model: nn.Module, dataloader: DataLoader, device: str, sgld_config: SGLDConfig) -> nn.Module:
    start_time = time.time()
    optimizer = OPTIMIZER_DICT[sgld_config.optimizer](model.parameters(), lr=sgld_config.learning_rate)
    criterion = CRITERION_DICT[sgld_config.criterion]
    
    for i in range(sgld_config.num_steps):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                new_params = torch.nn.utils.parameters_to_vector(model.parameters()) 
                noise = torch.randn_like(new_params) * sgld_config.temperature
                torch.nn.utils.vector_to_parameters(new_params + noise, model.parameters())

    print(f"LocalSGLD sampling took {time.time() - start_time:.2f} seconds")
    return model

def _sgld_parallel(models: List[nn.Module], dataloader: DataLoader, device: str, sgld_config: SGLDConfig) -> List[nn.Module]:
    start_time = time.time()
    optimizers = [OPTIMIZER_DICT[sgld_config.optimizer](model.parameters(), lr=sgld_config.learning_rate) for model in models]
    criterion = CRITERION_DICT[sgld_config.criterion]
    
    # Create separate dataloaders for each model to ensure different data points
    dataloaders = [
        DataLoader(
            dataloader.dataset,
            batch_size=sgld_config.batch_size,
            shuffle=True,  # Ensure different ordering for each model
            num_workers=0  # Avoid potential issues with multiple workers
        ) for _ in range(len(models))
    ]
    
    for i in range(sgld_config.num_steps):
        # Get different batches for each model
        data_target_pairs = [next(iter(dl)) for dl in dataloaders]
        
        # Process all models in parallel with their unique data
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        # Forward pass for all models with their unique data
        outputs = [model(data.to(device)) for model, (data, _) in zip(models, data_target_pairs)]
        losses = [criterion(output, target.to(device)) for output, (_, target) in zip(outputs, data_target_pairs)]
        
        # Backward pass for all models
        for loss in losses:
            loss.backward()
        
        # Update all models
        for optimizer in optimizers:
            optimizer.step()
        
        # Add noise to all models
        with torch.no_grad():
            for model in models:
                new_params = torch.nn.utils.parameters_to_vector(model.parameters())
                noise = torch.randn_like(new_params) * sgld_config.temperature
                torch.nn.utils.vector_to_parameters(new_params + noise, model.parameters())

    print(f"Parallel SGLD sampling took {time.time() - start_time:.2f} seconds")
    return models

class AutoencoderDataset(Dataset):
    def __init__(self, data: List[tuple]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def _create_vectorized_model_function(model: nn.Module, dataloader: DataLoader, device: str) -> torch.Tensor:
    out = []
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            out.append(logits.detach().flatten())  # Detach the tensor
    return torch.cat(out)

def _create_mlp_to_vectorized_model_function_dataset(mlp: nn.Module, dataloader: DataLoader, sgld_config: SGLDConfig, n_models: int, train_eval_split: float, device: str, dir_path: Optional[pathlib.Path] = None) -> Tuple[AutoencoderDataset, AutoencoderDataset, DataLoader, DataLoader]:
    start_time = time.time()
    # Create n_models copies of the MLP
    models = [MLP(mlp.config).to(device) for _ in range(n_models)]
    # Copy weights from the original MLP to all copies
    for model in models:
        model.load_state_dict(mlp.state_dict())
    
    # Run parallel SGLD
    models = _sgld_parallel(models, dataloader, device, sgld_config)
    print(f"Total SGLD sampling took {time.time() - start_time:.2f} seconds")
    
    autoencoder_training_data = [(model.flatten(), _create_vectorized_model_function(model, dataloader, device)) for model in models]

    train_size = int(train_eval_split * len(autoencoder_training_data))
    eval_size = len(autoencoder_training_data) - train_size
    train_data, eval_data = torch.utils.data.random_split(
        autoencoder_training_data, 
        [train_size, eval_size]
    )

    if dir_path is not None:
        torch.save(train_data, dir_path / "autoencoder_train_dataset.pt")
        torch.save(eval_data, dir_path / "autoencoder_eval_dataset.pt")

    return AutoencoderDataset(train_data), AutoencoderDataset(eval_data)


def _evaluate_autoencoder(autoencoder: nn.Module, train_dataloader: DataLoader, eval_dataloader: DataLoader, num_labels: int, device: str) -> float: # TODO: Add a parameter for top logit vs. full 
    def _evaluate_autoencoder_on_dataloader(autoencoder: nn.Module, dataloader: DataLoader, num_labels: int, device: str) -> dict: # TODO: Create a dataclass for evals
        out = dict(correct=[], incorrect=[])
        accuracies = []
        autoencoder.eval()
        with torch.no_grad():
            for idx, (model, ground_truth_vector_model_function) in enumerate(dataloader):
                local_out = dict(model=model, correct=[], incorrect=[])
                model = model.to(device)
                predicted_vector_model_function = autoencoder(model)
                
                predicted_vector_model_function = predicted_vector_model_function[0] # hotfix for now
                ground_truth_vector_model_function = ground_truth_vector_model_function[0] # hotfix for now
                
                for i in range(int(ground_truth_vector_model_function.shape[0] / num_labels)): # hotfix for now
                    ground_truth_label = ground_truth_vector_model_function[i * num_labels:(i + 1) * num_labels].argmax()
                    predicted_label = predicted_vector_model_function[i * num_labels:(i + 1) * num_labels].argmax()

                    if ground_truth_label == predicted_label:
                        local_out["correct"].append(predicted_vector_model_function)
                    else:
                        local_out["incorrect"].append(predicted_vector_model_function)
                
                local_accuracy = len(local_out['correct']) / (len(local_out['incorrect']) + len(local_out['correct']))
                print(f" Model {idx} reconstructed with {local_accuracy} accuracy")
                local_out["accuracy"] = local_accuracy
                accuracies.append(local_accuracy)
            out["accuracies"] = sum(accuracies) / len(accuracies)
            return out
    
    train_out = _evaluate_autoencoder_on_dataloader(autoencoder, train_dataloader, num_labels, device)
    print(f"Train accuracy: {train_out['accuracies']}")
    eval_out = _evaluate_autoencoder_on_dataloader(autoencoder, eval_dataloader, num_labels, device)
    print(f"Eval accuracy: {eval_out['accuracies']}")

    return train_out, eval_out


def main(
    mlp_config: MLPConfig,
    autoencoder_config: AutoencoderConfig,
    mlp_train_config: TrainConfig,
    autoencoder_train_config: TrainConfig,
    sgld_config: SGLDConfig,
    n_models: int,
    n_devices: int = 1,
    train_eval_split: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # TODO: too many device parameters being passed around
):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # mean and variance of MNIST
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=mlp_train_config.batch_size, shuffle=True)

    eval_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=mlp_train_config.batch_size, shuffle=True)
    
    mlp = MLP(mlp_config)
    mlp.optimize(mlp_train_config, train_dataloader, eval_dataloader)
    
    autoencoder_train_dataset, autoencoder_eval_dataset = _create_mlp_to_vectorized_model_function_dataset(mlp, train_dataloader, sgld_config, n_models, train_eval_split, device)
    autoencoder_train_dataloader = DataLoader(
        autoencoder_train_dataset,
        batch_size=autoencoder_train_config.batch_size,
        shuffle=True
    )
    autoencoder_eval_dataloader = DataLoader(
        autoencoder_eval_dataset,
        batch_size=autoencoder_train_config.batch_size,
        shuffle=False  # No need to shuffle evaluation data
    )

    autoencoder = Autoencoder(autoencoder_config)
    autoencoder.optimize(autoencoder_train_config, autoencoder_train_dataloader, test_loader=autoencoder_eval_dataloader, eval_metric="loss")
    _evaluate_autoencoder(autoencoder, autoencoder_train_dataloader, autoencoder_eval_dataloader, num_labels=10, device=device)

if __name__ == "__main__":
    mlp_config = MLPConfig(
        input_dim=784,
        hidden_layers=[128, 64],
        output_dim=10
    )
    
    mlp_train_config = TrainConfig(
        epochs=1,
        batch_size=64,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="cross_entropy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    autoencoder_config = AutoencoderConfig(
        input_dim=101770,  # total_mlp_params
        encoder_layers=[
            {"dim": 512, "activation": "relu"},
            {"dim": 256, "activation": "relu"},
            {"dim": 128, "activation": "relu"}
        ],
        decoder_layers=[
            {"dim": 256, "activation": "relu"},
            {"dim": 512, "activation": "relu"},
            {"dim": 7e5, "activation": "id"} # dataset_size x labels
        ]
    )
    
    autoencoder_train_config = TrainConfig(
        epochs=5,
        batch_size=10,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="mse",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    sgld_config = SGLDConfig(
        learning_rate=0.001,
        batch_size=64,
        criterion="cross_entropy",
        optimizer="sgd",
        temperature=0.01,
        num_steps=2
    )

    main(mlp_config, autoencoder_config, mlp_train_config, autoencoder_train_config, sgld_config, n_models=100, n_devices=10)