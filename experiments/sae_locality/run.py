from pathlib import Path
from typing import Optional
import einops
import fire
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from qft_nn.nn.sae import AutoEncoder, AutoEncoderConfig, extract_toy_model_activations
from qft_nn.nn.toy_model import SingleLayerToyReLUModelConfig, SingleLayerToyReLUModel
from qft_nn.ntk import compute_empirical_ntk

def _train_ground_truth_model(cfg: SingleLayerToyReLUModelConfig, device: torch.types.Device) -> SingleLayerToyReLUModel:
    # importance varies within features for each instance
    importance = (0.9 ** torch.arange(cfg.n_features))
    importance = einops.rearrange(importance, "features -> () features")

    # sparsity is the same for all features in a given instance, but varies over instances
    feature_probability = (50 ** -torch.linspace(0, 1, cfg.train.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = SingleLayerToyReLUModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=10_000)

    return model

def _plot_heatmap(matrix: torch.Tensor, title: str, figsize: int=(10,8), cmap: str = "Blue", annot: bool=True, fmt="2.f"):
    matrix = matrix.detach().numpy()
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap using seaborn
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    
    # If it's a 2D matrix, add row and column indices
    if len(matrix.shape) == 2:
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
    
    plt.tight_layout()
    plt.show()
    
def main(toy_model_cfg_path: Path, sae_cfg_path: Path, device: Optional[torch.types.Device]):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    toy_model_cfg = SingleLayerToyReLUModelConfig.from_yaml(filepath=toy_model_cfg_path)
    toy_model = _train_ground_truth_model(toy_model_cfg, device)
    
    sae_cfg = AutoEncoderConfig.from_yaml(filepath=sae_cfg_path)
    sae = AutoEncoder(cfg=sae_cfg, device=device)
    sae.optimize()
    
    data=toy_model.generate_batch(25)
    entk = compute_empirical_ntk(model=toy_model, loss=toy_model.calculate_loss, data=data)
    
    sae_matrix = extract_toy_model_activations(toy_model=toy_model, sae=sae, batch_size=toy_model_cfg.train.batch_size)
    pseudo_inverse = torch.linalg.pinv(sae_matrix.transpose())
    hypothesis_matrix = pseudo_inverse @ entk @ pseudo_inverse


    #Results
    frobenius_distance = torch.norm(entk - hypothesis_matrix, p='fro')
    print(f"Frobenius Distance between matrices: {frobenius_distance}")
    
    _plot_heatmap(matrix=entk, title="Empirical NTK")
    _plot_heatmap(matrix=hypothesis_matrix, title="(SAEact^T)^{P} ENTK(x,x’) SAEact^{P}")    

if __name__ == "__main__":
    # fire.Fire(main)
    from qft_nn.nn.base_config import TrainConfig
    from qft_nn.nn.lr_schedules import constant_lr

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
        n_input_ae = 32,
        n_hidden_ae = 64,
        l1_coeff = 1.0,
        tied_weights = False,
        train=train_config,
    )

    toy_model = SingleLayerToyReLUModel(cfg=toy_model_config, device='cpu')
    toy_model.optimize(steps=10_000)
    
    sae_cfg = AutoEncoderConfig.from_yaml(filepath=sae_config)
    sae = AutoEncoder(cfg=sae_cfg, device='cpu')
    sae.optimize()
    
    data=toy_model.generate_batch(25)
    entk = compute_empirical_ntk(model=toy_model, loss=toy_model.calculate_loss, data=data)
    
    sae_matrix = extract_toy_model_activations(toy_model=toy_model, sae=sae, batch_size=toy_model_config.train.batch_size)
    pseudo_inverse = torch.linalg.pinv(sae_matrix.transpose())
    hypothesis_matrix = pseudo_inverse @ entk @ pseudo_inverse

    #Results
    frobenius_distance = torch.norm(entk - hypothesis_matrix, p='fro')
    print(f"Frobenius Distance between matrices: {frobenius_distance}")
    
    _plot_heatmap(matrix=entk, title="Empirical NTK")
    _plot_heatmap(matrix=hypothesis_matrix, title="(SAEact^T)^{P} ENTK(x,x’) SAEact^{P}")    
