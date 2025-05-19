from typing import List, Literal, Tuple, Optional, Dict
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass
import time
import os

from qft_nn.models import MLP, TopKDictionary, TopKDictionaryConfig, MLPConfig, TrainConfig, OPTIMIZER_DICT, CRITERION_DICT, Optimizer, Criterion
from qft_nn.sgld import _sgld_parallel, SGLDConfig
from qft_nn.dataset import _create_vectorized_model_function, _create_mlp_to_vectorized_model_function_dataset, DictionaryDataset

DatasetMetrics = Literal["weight", "function", "label_projected", "loss"]
def _analyze_dataset(seed_model: nn.Module, dataset: Dataset, device: str, metrics: List[DatasetMetrics]) -> Dict[DatasetMetrics, List[float]]:
    raise NotImplementedError("This function is not implemented")
    out = dict()
    for x, y in dataset:
        x, y = x.to(device), y.to(device)
        if "weight" in metrics:
            out["weight"].append(abs(x - seed_model.weight.detach().cpu().numpy()))
        if "function" in metrics:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            out["function"].append(abs(y - _create_vectorized_model_function(seed_model, dataset, device)))
        if "label_projected" in metrics:
            out["label_projected"].append(seed_model(x).argmax(dim=-1).detach().cpu().numpy())
        if "loss" in metrics:
            out["loss"].append(seed_model(x).detach().cpu().numpy())
    
    return out


def _evaluate_dictionary(autoencoder: nn.Module, train_dataloader: DataLoader, eval_dataloader: DataLoader, num_labels: int, device: str) -> float: # TODO: Add a parameter for top logit vs. full 
    def _evaluate_dictionary_on_dataloader(autoencoder: nn.Module, dataloader: DataLoader, num_labels: int, device: str) -> dict: # TODO: Create a dataclass for evals
        out = dict(correct=[], incorrect=[])
        accuracies = []
        autoencoder.eval()
        with torch.no_grad():
            for batch_idx, (model_batch, ground_truth_batch) in enumerate(dataloader):
                local_out = dict(model=model_batch, correct=[], incorrect=[])
                model_batch = model_batch.to(device)
                predicted_batch = autoencoder(model_batch)
                
                # Process each model in the batch
                for model_idx in range(len(model_batch)):
                    model_predictions = predicted_batch[model_idx]
                    model_ground_truth = ground_truth_batch[model_idx]
                    
                    # Process each sample's predictions
                    num_samples = model_ground_truth.shape[0] // num_labels
                    correct_count = 0
                    
                    for sample_idx in range(num_samples):
                        start_idx = sample_idx * num_labels
                        end_idx = (sample_idx + 1) * num_labels
                        
                        ground_truth_label = model_ground_truth[start_idx:end_idx].argmax()
                        predicted_label = model_predictions[start_idx:end_idx].argmax()

                        if ground_truth_label == predicted_label:
                            local_out["correct"].append(model_predictions)
                            correct_count += 1
                        else:
                            local_out["incorrect"].append(model_predictions)
                    
                    local_accuracy = correct_count / num_samples
                    print(f" Model {batch_idx * len(model_batch) + model_idx} reconstructed with {local_accuracy:.4f} accuracy")
                    local_out["accuracy"] = local_accuracy
                    accuracies.append(local_accuracy)
                    
            out["accuracies"] = sum(accuracies) / len(accuracies)
            return out
    
    train_out = _evaluate_dictionary_on_dataloader(autoencoder, train_dataloader, num_labels, device)
    print(f"Train accuracy: {train_out['accuracies']}")
    eval_out = _evaluate_dictionary_on_dataloader(autoencoder, eval_dataloader, num_labels, device)
    print(f"Eval accuracy: {eval_out['accuracies']}")

    return train_out, eval_out

def main(
    mlp_config: MLPConfig,
    topk_dictionary_config: TopKDictionaryConfig,
    mlp_train_config: TrainConfig,
    topk_dictionary_train_config: TrainConfig,
    sgld_config: SGLDConfig,
    n_models: int,
    n_devices: int = 1,
    train_eval_split: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # TODO: too many device parameters being passed around
):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        transforms.Lambda(lambda x: x.flatten())  # Flatten the image
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=mlp_train_config.batch_size, shuffle=True)

    eval_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=mlp_train_config.batch_size, shuffle=True)
    
    mlp = MLP(mlp_config)
    mlp.optimize(mlp_train_config, train_dataloader, eval_dataloader)
    
    dictionary_train_dataset, dictionary_eval_dataset = _create_mlp_to_vectorized_model_function_dataset(mlp, train_dataloader, sgld_config, n_models, train_eval_split, device)
    dictionary_train_dataloader = DataLoader(
        dictionary_train_dataset,
        batch_size=topk_dictionary_train_config.batch_size,
        shuffle=True
    )
    dictionary_eval_dataloader = DataLoader(
        dictionary_eval_dataset,
        batch_size=topk_dictionary_train_config.batch_size,
        shuffle=False  # No need to shuffle evaluation data
    )

    topk_dictionary = TopKDictionary(topk_dictionary_config)
    topk_dictionary.optimize(topk_dictionary_train_config, dictionary_train_dataloader, dictionary_eval_dataloader)
    _evaluate_dictionary(topk_dictionary, dictionary_train_dataloader, dictionary_eval_dataloader, num_labels=10, device=device)

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

    topk_dictionary_config = TopKDictionaryConfig(
        input_dim=101770,
        latent_dim=128,
        output_dim=600000,
        k=10,
        activation="relu",
        encoder_bias=True,
        decoder_bias=True,
        input_center=None,
        output_center=None
    )
    
    topk_dictionary_train_config = TrainConfig(
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

    main(mlp_config, topk_dictionary_config, mlp_train_config, topk_dictionary_train_config, sgld_config, n_models=10, n_devices=10)
