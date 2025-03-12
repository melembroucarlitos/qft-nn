import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

from qft_nn.ntk import compute_empirical_ntk

def test_compute_empirical_ntk():
    # TODO: Incorrect type signatures
    # Simple Linear Model, MSE loss

    class SimpleLinear(nn.Module):
        def __init__(self, input_dim=2, output_dim=1):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(0.5)  # All weights = 0.5
                self.linear.bias.fill_(0.1)    # Bias = 0.1
        
        def forward(self, x):
            return self.linear(x)

    # Create a very simple dataset: 3 points in 2D
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.5], [0.3], [0.7]], dtype=torch.float32)
    dataset = TensorDataset(X, y)
    
    # Create the model
    model = SimpleLinear(input_dim=2, output_dim=1)
    criterion = nn.MSELoss()
    
    # Manual Calculations for ground truth NTK for this simple linear model with MSE loss

    # Claim: For MSE loss: gradient w.r.t parameters = 2*(pred - target)*[input, 1]
    # Justification:
    #   The MSE loss is: L = (f(x) - y)² = (w·x + b - y)²
    #   For weights w:
    #     ∂L/∂w = 2(w·x + b - y) · ∂(w·x + b)/∂w (chain rule)
    #     ∂(w·x + b)/∂w = x
    #     Thus: ∂L/∂w = 2(pred - target) · x
    #   For bias b:
    #     ∂L/∂b = 2(w·x + b - y) · ∂(w·x + b)/∂b
    #     ∂(w·x + b)/∂b = 1
    #     Thus: ∂L/∂b = 2(pred - target) · 1
    #   Combined as one vector: ∂L/∂[w,b] = 2(pred - target)*[x, 1]

    # Claim: ENTK(x,x') = 4*(pred_i - target_i)*(pred_j - target_j)*(x·x' + 1)
    # Justification:
    #   Given that ENTK is the dot product of parameter gradients:
    #   ENTK(x_i,x_j) = ∇[w,b]L(x_i)·∇[w,b]L(x_j)
    #   For data points x_i and x_j:
    #     ∇[w,b]L(x_i) = 2*(pred_i - target_i)*[x_i, 1]
    #     ∇[w,b]L(x_j) = 2*(pred_j - target_j)*[x_j, 1]
    #   Taking the dot product:
    #   ENTK(x_i,x_j) = (2*(pred_i - target_i)*[x_i, 1])·(2*(pred_j - target_j)*[x_j, 1])
    #                  = 4*(pred_i - target_i)*(pred_j - target_j)*([x_i, 1]·[x_j, 1])
    #                  = 4*(pred_i - target_i)*(pred_j - target_j)*(x_i·x_j + 1)

    ground_truth_entk = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            error_i = model.forward(X[i]) - y[i].item()
            error_j = model.forward(X[j]) - y[j].item()
            feature_similarity = torch.dot(X[i], X[j]).item()

            ground_truth_entk[i, j] = 4 * error_i * error_j * (feature_similarity + 1.0)
    
    # Compute empirical NTK
    entk = compute_empirical_ntk(model, criterion, dataset)

    # Check if computed NTK matches ground truth
    np.testing.assert_allclose(entk, ground_truth_entk, rtol=1e-5)

    # TODO: 2 Layer MLP, MSE loss

if __name__ == "__main__":
    test_compute_empirical_ntk()