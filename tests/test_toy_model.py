import pathlib
import tempfile
import pytest

from qft_nn.nn.toy_model import SingleLayerToyReLUModelConfig 

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