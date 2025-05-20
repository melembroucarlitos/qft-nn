import torch
from torch.utils.data import DataLoader
import pathlib
import pickle
from typing import Optional

from qft_nn.models import MLP, MLPConfig, TrainConfig, TopKDictionary, TopKDictionaryConfig
from qft_nn.dataset import DictionaryDataset, _create_mnist_dataloaders, _create_vectorized_model_function, _create_and_save_dataset
from qft_nn.analysis import _evaluate_dictionary, _distance_from_seed_model_dictionary_dataset_distribution_metrics, _plot_dataset_metrics, DictionaryDatasetAnalysisConfig, DictionaryAnalysisConfig
from qft_nn.dataset import DatasetGenerationConfig
from qft_nn.sgld import SGLDConfig

def main(
    seed_model_config: MLPConfig,
    seed_model_train_config: Optional[TrainConfig] = None,
    seed_model_file_path: Optional[pathlib.Path] = None,
    dictionary_dataset_generation_config: Optional[DatasetGenerationConfig] = None,
    dictionary_dataset_analysis_config: Optional[DictionaryDatasetAnalysisConfig] = None,
    dictionary_dataset_train_file_path: Optional[pathlib.Path] = None,
    dictionary_dataset_eval_file_path: Optional[pathlib.Path] = None,
    topk_dictionary_train_config: Optional[TrainConfig] = None,
    topk_dictionary_file_path: Optional[pathlib.Path] = None,
    topk_dictionary_analysis_config: Optional[DictionaryAnalysisConfig] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    
    if seed_model_train_config is not None or topk_dictionary_train_config is not None:
        mnist_train_dataloader, mnist_eval_dataloader = _create_mnist_dataloaders(batch_size=10, shuffle=True)

    assert seed_model_file_path is not None or seed_model_train_config is not None, "Either seed_model_file_path or seed_model_train_config must be provided"
    assert not (seed_model_file_path is not None and seed_model_train_config is not None), "Only one of seed_model_file_path or seed_model_train_config must be provided"

    # Either load or train a seed_model
    mlp = MLP(seed_model_config)
    if seed_model_file_path is not None:
        mlp.load_state_dict(torch.load(seed_model_file_path))
    elif seed_model_train_config is not None:
        mlp.optimize(
            config=seed_model_train_config, 
            train_loader=mnist_train_dataloader, 
            test_loader=mnist_eval_dataloader, 
            model_save_path=seed_model_train_config.model_save_path,
            config_save_path=seed_model_train_config.config_save_path
        )
    
    mlp = mlp.to(device)

    dictionary_dataset_file_paths = dictionary_dataset_train_file_path is not None and dictionary_dataset_eval_file_path is not None
    assert dictionary_dataset_file_paths or dictionary_dataset_generation_config is not None, "Either dictionary_dataset_file_path or dictionary_dataset_generation_config must be provided"
    assert not (dictionary_dataset_file_paths and dictionary_dataset_generation_config is not None), "Only one of dictionary_dataset_file_path or dictionary_dataset_generation_config must be provided"

    # Either load or generate a dataset
    if dictionary_dataset_file_paths:
        dictionary_train_dataset = DictionaryDataset.load_from_file(dictionary_dataset_train_file_path)
        dictionary_eval_dataset = DictionaryDataset.load_from_file(dictionary_dataset_eval_file_path)
    elif dictionary_dataset_generation_config is not None:
        dictionary_train_dataset, dictionary_eval_dataset =  _create_and_save_dataset( # TODO: Take saving out of this function
            seed_model=mlp,
            sgld_config=dictionary_dataset_generation_config.sgld_config,
            n_models=dictionary_dataset_generation_config.n_models,
            train_eval_split=dictionary_dataset_generation_config.train_eval_split,
            dir_path=dictionary_dataset_generation_config.output_dir,
            device=device
        ) # TODO: Don't pass in configs, pass in parameters from configs
        
    metrics = _distance_from_seed_model_dictionary_dataset_distribution_metrics(
        seed_model=mlp, 
        seed_model_train_dataloader=mnist_train_dataloader, 
        dictionary_dataset=dictionary_train_dataset, 
        distance_function=dictionary_dataset_analysis_config.distance_function, 
        device=device, metrics=dictionary_dataset_analysis_config.metrics, 
        labels=dictionary_dataset_analysis_config.labels
    )

    # TODO: Pull the save_dir out from _plot_dataset_metrics
    _plot_dataset_metrics(
        metrics=metrics, 
        distance_function=dictionary_dataset_analysis_config.distance_function, 
        title="Distance from seed model", 
        xlabel="Distance", 
        ylabel="Frequency", 
        save_dir=dictionary_dataset_analysis_config.plots_save_dir, 
        show=True
    )
    
    assert topk_dictionary_file_path is not None or topk_dictionary_train_config is not None, "Either topk_dictionary_file_path or topk_dictionary_train_config must be provided"
    assert not (topk_dictionary_file_path is not None and topk_dictionary_train_config is not None), "Only one of topk_dictionary_file_path or topk_dictionary_train_config must be provided"

    topk_dictionary = TopKDictionary(topk_dictionary_config)
    
    if topk_dictionary_file_path is not None:
        topk_dictionary.load_state_dict(torch.load(topk_dictionary_file_path))
    elif topk_dictionary_train_config is not None:
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

        topk_dictionary_config.input_center = mlp.flatten()
        topk_dictionary_config.output_center = _create_vectorized_model_function(
            model=mlp, 
            dataloader=mnist_train_dataloader, 
            device=device
        )
        topk_dictionary.optimize(
            config=topk_dictionary_train_config, 
            train_loader=dictionary_train_dataloader, 
            test_loader=dictionary_eval_dataloader,
            model_save_path=topk_dictionary_train_config.model_save_path,
            config_save_path=topk_dictionary_train_config.config_save_path
        )

    topk_dictionary.to(device)

    # Evaluate the topk dictionary
    topk_evals = _evaluate_dictionary(
        topk_dictionary=topk_dictionary, 
        dictionary_train_dataloader=dictionary_train_dataloader, 
        dictionary_eval_dataloader=dictionary_eval_dataloader, 
        num_labels=topk_dictionary_analysis_config.num_labels, 
        device=device
    )

    with open(topk_dictionary_analysis_config.output_dir / "topk_evals.pkl", "wb") as f:
        pickle.dump(topk_evals, f)

    # TODO: Include a remote persistence config & scripts


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
        model_save_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/seed_mlp_model.pt"),
        config_save_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/seed_mlp_model_train_config.json"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    topk_dictionary_train_config = TrainConfig(
        epochs=5,
        batch_size=10,
        learning_rate=0.001,
        eval_every_n_batches=100,
        optimizer="adam",
        criterion="mse",
        model_save_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/topk_dictionary.pt"),
        config_save_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/topk_dictionary_train_config.json"),
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
    
    topk_dictionary_config = TopKDictionaryConfig(
        input_dim=101770,
        latent_dim=128,
        output_dim=600000,
        k=10,
        activation="relu",
        encoder_bias=True,
        decoder_bias=False,
        input_center=None,
        output_center=None,
    )

    # Generate dataset configs
    dataset_generation_config = DatasetGenerationConfig(
        mlp_config=mlp_config,
        mlp_train_config=mlp_train_config,
        sgld_config=sgld_config,
        n_models=2, # TODO: Make a check that this is at least 2
        train_eval_split=0.8,
        output_dir=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/")
    )

    dictionary_dataset_analysis_config = DictionaryDatasetAnalysisConfig(
        plots_save_dir=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/plots"),
        metrics=["weight", "function"],
        distance_function="l2",
        labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    )

    # Generate evaluation configs 
    topk_dictionary_analysis_config = DictionaryAnalysisConfig(
        num_labels=10,
        output_dir=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/plots")
    )

    # Run main with all configs
    main(
        seed_model_config=mlp_config,
        # seed_model_train_config=mlp_train_config,
        seed_model_file_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/artifacts/seed_mlp_model.pt"),
        dictionary_dataset_generation_config=dataset_generation_config,
        # dictionary_dataset_train_file_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/results/dictionary_train.pth"),
        # dictionary_dataset_eval_file_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/results/dictionary_eval.pth"),
        dictionary_dataset_analysis_config=dictionary_dataset_analysis_config,
        topk_dictionary_train_config=topk_dictionary_train_config,
        # topk_dictionary_file_path=pathlib.Path("experiments/topkdictionary_mlp_mnist_bottleneck/results/topk_dictionary.pth"),
        topk_dictionary_analysis_config=topk_dictionary_analysis_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )