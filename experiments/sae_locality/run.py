from pathlib import Path
from typing import Optional
import einops
import fire
import torch

from qft_nn.toy_model import SingleLayerToyReLUModelConfig, SingleLayerToyReLUModel

def _train_ground_truth_model(cfg: SingleLayerToyReLUModelConfig, device: torch.types.Device) -> SingleLayerToyReLUModel:
    # importance varies within features for each instance
    importance = (0.9 ** torch.arange(cfg.n_features))
    importance = einops.rearrange(importance, "features -> () features")

    # sparsity is the same for all features in a given instance, but varies over instances
    feature_probability = (50 ** -torch.linspace(0, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = SingleLayerToyReLUModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=10_000)

    return model
    
def main(cfg_path: Path, device: Optional[torch.types.Device]):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = SingleLayerToyReLUModelConfig.from_yaml(filepath=cfg_path)
    
    toy_model = _train_ground_truth_model(cfg, device)
    raise NotImplementedError
    # Run ENTK on ground truth model
    # Train SAEs on ground truth model
    # Compare Frobenius Norm
    # ?? Shuffle Control ??

if __name__ == "__main__":
    fire.Fire(main)