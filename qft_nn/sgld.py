from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List
from torch.utils.data import DataLoader

from qft_nn.models import Optimizer, Criterion, OPTIMIZER_DICT, CRITERION_DICT

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
                noise = torch.randn_like(new_params) * np.sqrt(sgld_config.learning_rate * sgld_config.temperature)
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