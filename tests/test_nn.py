import pathlib
import torch
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from qft_nn.nn.base_config import TrainConfig
from qft_nn.nn.lr_schedules import constant_lr
from qft_nn.nn.toy_model import SingleLayerToyReLUModelConfig, SingleLayerToyReLUModel

# Test Config.from_yaml

def test_from_yaml():
    # Create a temporary YAML file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        temp_file.write("""
        n_instances: 100
        n_features: 10
        n_hidden: 5
        n_correlated_pairs: 2
        n_anticorrelated_pairs: 3
        """)
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # Load the config from the YAML file
        config = SingleLayerToyReLUModelConfig.from_yaml(temp_path)
        
        # Verify the config values
        assert isinstance(config, SingleLayerToyReLUModelConfig)
        assert config.n_instances == 100
        assert config.n_features == 10
        assert config.n_hidden == 5
        assert config.n_correlated_pairs == 2
        assert config.n_anticorrelated_pairs == 3
    finally:
        # Clean up the temporary file
        temp_path.unlink()

def test_from_yaml_extra_fields():
    # Test that extra fields raise an error due to extra="forbid"
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        temp_file.write("""
        n_instances: 100
        n_features: 10
        n_hidden: 5
        n_correlated_pairs: 2
        n_anticorrelated_pairs: 3
        extra_field: "This should cause an error"
        """)
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # This should raise a ValidationError
        with pytest.raises(Exception):  # You might want to use a more specific exception here
            SingleLayerToyReLUModelConfig.from_yaml(temp_path)
    finally:
        # Clean up the temporary file
        temp_path.unlink()

def test_from_yaml_missing_fields():
    # Test that missing required fields raise an error
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        temp_file.write("""
        n_instances: 100
        n_features: 10
        # Missing n_hidden, n_correlated_pairs, and n_anticorrelated_pairs
        """)
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # This should raise a ValidationError
        with pytest.raises(Exception):  # You might want to use a more specific exception here
            SingleLayerToyReLUModelConfig.from_yaml(temp_path)
    finally:
        # Clean up the temporary file
        temp_path.unlink()


# Test Optimization

@pytest.fixture
def model():
    # Create a config for our model
    config = SingleLayerToyReLUModelConfig(
        n_instances=2,
        n_features=10,
        n_hidden=5,
        n_correlated_pairs=0,
        n_anticorrelated_pairs=0
    )
    
    # Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return SingleLayerToyReLUModel(cfg=config, device=device)

def test_optimize_decreases_loss(model):
    """Test that the optimization process decreases the loss over time."""
    # Run optimization for a small number of steps
    with patch('qft_nn.nn.toy_model.tqdm') as mock_tqdm:
        mock_progress = MagicMock()
        mock_tqdm.return_value = mock_progress
        mock_progress.__iter__.return_value = range(50)
        
        # Capture loss values during optimization
        losses = []
        
        def custom_set_postfix(**kwargs):
            losses.append(kwargs['loss'])
        
        mock_progress.set_postfix.side_effect = custom_set_postfix
        
        # Run optimization
        model.optimize(batch_size=32, steps=50, log_freq=10, lr=0.01)
        
        # Verify tqdm was called correctly
        mock_tqdm.assert_called_once()
        
        # Check that we have some loss values recorded
        assert len(losses) > 0
        
        # Check that loss decreased
        assert losses[-1] < losses[0]

def test_conditional_determinism_in_data_generation():
    train_config = TrainConfig(
        n_instances=1,
        batch_size=1024,
        steps=10_000,
        log_freq=100,
        lr=1e-3,
        lr_scale=constant_lr,
        data_seed=1337
    )

    model_config = SingleLayerToyReLUModelConfig(
        n_features=5,
        n_hidden=32,
        n_correlated_pairs=2,
        n_anticorrelated_pairs=1,
        train=train_config
    )

    # Instantiate the model
    model = SingleLayerToyReLUModel(cfg=model_config, device='cpu')

    batch_size = 1000
    
    batch1 = model.generate_batch(batch_size=batch_size, deterministic=True)
    batch2 = model.generate_batch(batch_size=batch_size, deterministic=True)
    
    # Check if they're identical
    assert torch.all(batch1 == batch2).item()

    batch2 = model.generate_batch(batch_size=batch_size, deterministic=False)

    assert not torch.all(batch1 == batch2).item()

