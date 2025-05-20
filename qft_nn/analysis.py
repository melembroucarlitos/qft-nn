from typing import List, Literal, Tuple, Optional, Dict
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass
from jaxtyping import Float
import matplotlib.pyplot as plt

from qft_nn.models import MLP, TopKDictionary, TopKDictionaryConfig, MLPConfig, TrainConfig, OPTIMIZER_DICT, CRITERION_DICT, Optimizer, Criterion
from qft_nn.sgld import _sgld_parallel, SGLDConfig
from qft_nn.dataset import _create_vectorized_model_function, _create_mlp_to_vectorized_model_function_dataset, DictionaryDataset, _create_mnist_dataloaders

DatasetMetrics = Literal["weight", "function", "label_projected", "loss"]
DISTANCE_FUNCTION_DICT = {
    "l2": lambda x, y: torch.norm(x - y, p=2),
    "l1": lambda x, y: torch.norm(x - y, p=1),
    "cosine": lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0),
}

class DictionaryDatasetAnalysisConfig(BaseModel):
    plots_save_dir: Optional[pathlib.Path] = None
    metrics: List[str] = ["weight", "function"]
    distance_function: str = "l2"
    labels: Optional[List[str]] = None

def _distance_from_seed_model_dictionary_dataset_distribution_metrics(
    seed_model: nn.Module, 
    seed_model_train_dataloader: DataLoader, 
    dictionary_dataset: DictionaryDataset, 
    device: str,
    distance_function: Literal["l2", "l1", "cosine"],
    metrics: List[DatasetMetrics],
    labels: Optional[List[int]] = None
) -> Dict[DatasetMetrics, Float[np.ndarray, "n_datapoints"]]:

    if "label_projected" in metrics and labels is None:
        raise ValueError("labels must be provided if label_projected is in metrics")
    
    seed_model_datapoint = (seed_model.flatten(), _create_vectorized_model_function(seed_model, seed_model_train_dataloader, device))
    distance_function = DISTANCE_FUNCTION_DICT[distance_function]

    out = dict(weight=[], function=[], loss=[], label_projected={0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []})
    
    if "weight" in metrics:
        out["weight"] = np.array([abs(distance_function(model_flattened.to(device), seed_model_datapoint[0])).cpu().item() for model_flattened, _ in dictionary_dataset])
    if "function" in metrics:
        out["function"] = np.array([abs(distance_function(label.to(device), seed_model_datapoint[1])).cpu().item() for _, label in dictionary_dataset])
    if "loss" in metrics:
        raise NotImplementedError("This function is not implemented")
    if "label_projected" in metrics:
        function_vector = out["function"].reshape(-1, 10)
        for label in labels:
            projected_function_vector = function_vector[:, label]
            out["label_projected"][label] = projected_function_vector

    return out


def _plot_dataset_metrics(metrics: Dict[DatasetMetrics, Float[np.ndarray, "n_datapoints"]], distance_function: Literal["l2", "l1", "cosine"], title: str, xlabel: str, ylabel: str, save_dir: Optional[pathlib.Path] = None, show: bool = False):
    if not show and save_dir is None:
        raise ValueError("At least one of show or save must be True")

    for metric, data in metrics.items():
        plt.hist(data, bins=100, density=True)
        plt.title(f"{metric} {distance_function} {title}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show:
            plt.show()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"{metric}_{distance_function}_{title}.png")
        plt.close()

class DictionaryAnalysisConfig(BaseModel):
    num_labels: int
    output_dir: pathlib.Path

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
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

    train_out = _evaluate_dictionary_on_dataloader(autoencoder, train_dataloader, num_labels, device)
    print(f"Train accuracy: {train_out['accuracies']}")
    eval_out = _evaluate_dictionary_on_dataloader(autoencoder, eval_dataloader, num_labels, device)
    print(f"Eval accuracy: {eval_out['accuracies']}")

    return train_out, eval_out

def _analyze_topk_dictionary_learned_circuits(topk_dictionary: TopKDictionary, device: str):
    raise NotImplementedError("This function is not implemented")

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

    # main(mlp_config, topk_dictionary_config, mlp_train_config, topk_dictionary_train_config, sgld_config, n_models=10, n_devices=10)
    
    # Load trained MLP model
    mlp = MLP(mlp_config)
    mlp.load_state_dict(torch.load("/home/lucas/qft-nn/temporary_datasets/mlp_state_dict.pt"))
    mlp = mlp.to("cuda" if torch.cuda.is_available() else "cpu")

    dictionary_train_dataset = DictionaryDataset.load_from_file(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/dictionary_train_dataset.pt"))
    dictionary_eval_dataset = DictionaryDataset.load_from_file(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/dictionary_eval_dataset.pt"))

    mnist_train_dataloader, mnist_eval_dataloader = _create_mnist_dataloaders(batch_size=10, shuffle=True)

    # metrics = _distance_from_seed_model_dictionary_dataset_distribution_metrics(seed_model=mlp, seed_model_train_dataloader=mnist_train_dataloader, dictionary_dataset=dictionary_train_dataset, distance_function="l2", device="cuda", metrics=["weight", "function"], labels=None)
    # _plot_dataset_metrics(metrics, distance_function="l2", title="Distance from seed model", xlabel="Distance", ylabel="Frequency", save_dir=pathlib.Path("/home/lucas/qft-nn/temporary_datasets/"), show=True)

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

    topk_dictionary_config = TopKDictionaryConfig(
        input_dim=101770,
        latent_dim=128,
        output_dim=600000,
        k=10,
        activation="relu",
        encoder_bias=True,
        decoder_bias=False,
        input_center=mlp.flatten(),
        output_center=_create_vectorized_model_function(mlp, mnist_train_dataloader, device="cuda")
    )

    topk_dictionary = TopKDictionary(topk_dictionary_config)
    topk_dictionary.load_state_dict(torch.load("/home/lucas/qft-nn/temporary_datasets/topk_dictionary_state_dict.pt"))
    topk_dictionary.to(device="cuda" if torch.cuda.is_available() else "cpu")
    # topk_dictionary.optimize(topk_dictionary_train_config, dictionary_train_dataloader, dictionary_eval_dataloader)
    # torch.save(topk_dictionary.state_dict(), pathlib.Path("/home/lucas/qft-nn/temporary_datasets/topk_dictionary_state_dict.pt"))

    topk_evals = _evaluate_dictionary(topk_dictionary, dictionary_train_dataloader, dictionary_eval_dataloader, num_labels=10, device="cuda")
    with open(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/topk_evals.pkl"), "wb") as f:
        pickle.dump(topk_evals, f)

    learned_circuits = topk_dictionary.decoder.weight.T
