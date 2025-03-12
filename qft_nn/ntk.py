import torch
from torch.utils.data import DataLoader
import numpy as np
from jaxtyping import Float
from typing import Optional


def compute_empirical_ntk(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    data: Float[torch.Tensor, "data_dim num_data"],
    batch_size: Optional[int] = None
) -> Float[np.ndarray, "num_data num_data"]:
    # Prepare data
    if batch_size is None:
        dataloader = DataLoader(data, batch_size=len(data))
        X, y = next(iter(dataloader))
    else:
        dataloader = DataLoader(data, batch_size=batch_size)
        X_batches, y_batches = [], []
        for X_batch, y_batch in dataloader:
            X_batches.append(X_batch)
            y_batches.append(y_batch)
        
        X = torch.cat(X_batches, dim=0)
        y = torch.cat(y_batches, dim=0)

    # Initialize Jacobian
    n_samples = X.shape[0]
    parameters = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in parameters)
    jacobian = torch.zeros(n_samples, n_params)
        
    model.train()
    for i in range(n_samples):
        x_i = X[i:i+1]
        y_i = y[i:i+1]

        model.zero_grad()
        y_pred = model(x_i)
        loss = criterion(y_pred, y_i)
        loss.backward()
        
        # Extract gradients
        idx = 0
        for param in parameters:
            grad_flatten = param.grad.flatten()
            jacobian[i, idx:idx+len(grad_flatten)] = grad_flatten
            idx += len(grad_flatten)
    
    # ENTK: K(x, x') = J(x) · J(x')^T
    entk_matrix = jacobian @ jacobian.T
    return entk_matrix.detach().numpy()