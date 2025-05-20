from pydantic import BaseModel
from abc import ABC
import pathlib
import yaml

class ExperimentConfig(BaseModel, ABC):
    def save(self, path: pathlib.Path):
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert model to dict and handle pathlib.Path objects
        # HOTFIX
        config_dict = self.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, pathlib.Path):
                config_dict[key] = str(value)
        
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f)

    @classmethod
    def load(cls, path: pathlib.Path):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
            # Convert string paths back to pathlib.Path objects
            # HOTFIX
            for key, value in config_dict.items():
                if isinstance(value, str) and key.endswith('_path') or key == 'save_dir':
                    config_dict[key] = pathlib.Path(value)
            return cls(**config_dict)

if __name__ == "__main__":
    from qft_nn.models import MLPConfig, TrainConfig
    from qft_nn.sgld import SGLDConfig
    import torch
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

    mlp_config.save(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/mlp_config.yaml"))
    mlp_train_config.save(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/mlp_train_config.yaml"))
    sgld_config.save(pathlib.Path("/home/lucas/qft-nn/temporary_datasets/sgld_config.yaml"))
