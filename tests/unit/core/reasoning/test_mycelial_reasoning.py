
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, mock_open
from core.reasoning.mycelial_reasoning import (
    MycelialReasoning, MycelialConfig, MycelialVQVAE
)

# Helpers
def create_indices(h1, h2, h3, h4):
    return np.array([h1, h2, h3, h4])

@pytest.fixture
def mycelian():
    config = MycelialConfig(
        num_heads=4,
        codebook_size=100,
        learning_rate=0.5,
        decay_rate=0.9,
        min_weight=0.1
    )
    return MycelialReasoning(config)

def test_initialization(mycelian):
    assert len(mycelian.graph) == 0
    assert mycelian.step == 0

def test_observation(mycelian):
    indices = create_indices(10, 20, 30, 40)
    mycelian.observe(indices)
    
    # Check that nodes are created: (header_idx, code)
    # 4 nodes should be created
    assert len(mycelian.graph) == 4
    
    # Check connection (10 at head 0) <-> (20 at head 1)
    node_a = (0, 10)
    node_b = (1, 20)
    
    # Bidirectional
    assert mycelian.graph[node_a][node_b] == 0.5
    assert mycelian.graph[node_b][node_a] == 0.5

def test_propagation_and_reasoning(mycelian):
    # Train pattern [10, 10, 10, 10]
    for _ in range(5):
        mycelian.observe([10, 10, 10, 10])
        
    # Test reasoning with partial input [10, 0, 0, 0] (0 is noise/unknown)
    # Head 0 has code 10. This should activate code 10 in other heads.
    
    # Manually check propagation first
    activation = mycelian.propagate([10, 0, 0, 0], steps=1)
    
    # Head 1 should have code 10 activated
    assert activation.get((1, 10), 0) > 0
    
    # Full reason
    res, acts = mycelian.reason([10, 0, 0, 0])
    
    # Should restore [10, 10, 10, 10] if weights are strong enough
    # With 5 observations, weights are 2.5 per edge.
    # [10] activates neighbors.
    assert res[1] == 10
    assert res[2] == 10
    assert res[3] == 10

def test_decay(mycelian):
    # Setup nodes
    node_a = (0, 1)
    node_b = (1, 2)
    mycelian.graph[node_a][node_b] = 0.15
    
    # decay rate 0.9. New weight = 0.135 (above min 0.1)
    mycelian.decay()
    assert node_b in mycelian.graph[node_a]
    assert pytest.approx(mycelian.graph[node_a][node_b], 0.001) == 0.135
    
    # decay again. 0.135 * 0.9 = 0.1215
    # decay again until < 0.1
    for _ in range(5):
        mycelian.decay()
        
    # Should be removed eventually
    if node_a in mycelian.graph:
        assert node_b not in mycelian.graph[node_a]

def test_persistence(mycelian):
    # Mock open and pickle
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.dump") as mock_pickle_dump:
            mycelian.save_state("dummy_path.pkl")
            mock_pickle_dump.assert_called_once()
            
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("pickle.load") as mock_pickle_load:
            # Mock loaded state
            mock_pickle_load.return_value = {
                'graph': {(0,1): {(1,2): 0.5}},
                'node_activation_counts': {},
                'total_observations': 10,
                'step': 5,
                'config': {'num_heads': 4}
            }
            with patch("pathlib.Path.exists", return_value=True):
                success = mycelian._load_state()
                assert success is True
                assert mycelian.total_observations == 10
                assert mycelian.graph[(0,1)][(1,2)] == 0.5

# Test Wrapper
@pytest.fixture
def mock_vqvae_model():
    model = MagicMock()
    # Mock forward pass output
    model.return_value = {'indices': torch.tensor([[1, 2, 3, 4]])}
    return model

@pytest.fixture
def wrapper(mock_vqvae_model, mycelian):
    # Patch imports that happen inside the wrapper if any, or class level imports
    # The file imports MonolithV13 at module level. We need to patch it BEFORE importing the module if possible,
    # but since we already imported it, we mock the class usage.
    
    # Since MycelialVQVAE is already imported, we rely on dependency injection in __init__
    wrapper = MycelialVQVAE(mock_vqvae_model)
    # Inject our pre-configured mycelial instance
    wrapper.mycelial = mycelian
    return wrapper

def test_wrapper_encode(wrapper, mock_vqvae_model):
    x = torch.randn(1, 384)
    indices = wrapper.encode(x)
    assert indices.shape == (1, 4)
    mock_vqvae_model.assert_called_once()

def test_wrapper_pipeline(wrapper):
    x = torch.randn(1, 384)
    result = wrapper.full_pipeline(x)
    
    assert 'original_indices' in result
    assert 'reasoned_indices' in result
    assert wrapper.mycelial.total_observations == 1
