
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from core.reasoning.neural_learner import V2Learner

@pytest.fixture
def mock_model_cls():
    with patch("core.reasoning.neural_learner.MonolithV13") as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.to.return_value = mock_instance # Fluent API
        mock_instance.parameters.return_value = []
        
        # Mock forward output
        mock_instance.return_value = {
            'reconstructed': torch.tensor([[0.0]]), # Dummy tensor
            'z_e': torch.tensor([[0.0]]),
            'z_q': torch.tensor([[0.0]])
        }
        
        # Mock decoder
        mock_instance.decoder.return_value = torch.tensor([[0.0]])
        
        # Mock quantizer (for ortho loss)
        mock_instance.quantizer = MagicMock()
        
        yield MockModel

@pytest.fixture
def mock_optimizer_cls():
    with patch("torch.optim.AdamW") as MockOptim:
        yield MockOptim

@pytest.fixture
def mock_losses():
    # Return tensors with requires_grad=True to support backward()
    with patch("core.reasoning.neural_learner.compute_vq_commitment_loss", return_value=torch.tensor(0.1, requires_grad=True)) as m_vq, \
         patch("core.reasoning.neural_learner.compute_orthogonal_loss", return_value=torch.tensor(0.1, requires_grad=True)) as m_ortho:
        yield m_vq, m_ortho

@pytest.fixture
def learner(mock_model_cls, mock_optimizer_cls):
    # Mock torch.cuda.is_available
    with patch("torch.cuda.is_available", return_value=False):
        # Mock init file system calls
        with patch("pathlib.Path.exists", return_value=False):
             learner = V2Learner(device="cpu")
             return learner

def test_initialization(learner, mock_model_cls):
    assert learner.device == "cpu"
    mock_model_cls.assert_called_once()
    assert learner.is_loaded is True # Defaults to True if file missing (start from scratch)

def test_learn(learner, mock_losses):
    vectors = [[0.1] * 384]
    
    # Mock instance method directly
    learner._save_history = MagicMock()
    
    # Needs to mock MSE loss as well since arguments are tensors
    with patch("torch.nn.functional.mse_loss", return_value=torch.tensor(0.5, requires_grad=True)):
         metrics = learner.learn(vectors)
             
    assert "total_loss" in metrics
    assert learner.optimizer.step.called
    assert learner.optimizer.zero_grad.called

    # Check history
    assert len(learner.history) == 1
    learner._save_history.assert_called_once()

def test_encode(learner):
    vectors = [[0.1] * 384]
    # Configure mock model z_q attribute return
    learner.model.return_value = {'z_q': torch.tensor([[1.0, 2.0]])}
    
    latents = learner.encode(vectors)
    assert latents.shape == (1, 2)
    assert latents[0][0] == 1.0

def test_decode(learner):
    latents = [[1.0, 2.0]]
    decoded = learner.decode(latents)
    # Mock decoder returns tensor([[0.0]])
    assert decoded.shape == (1, 1) # Based on fixture mock

def test_save_model(learner):
    # Mock model_path on the instance to allow checking parent.mkdir
    learner.model_path = MagicMock()
    with patch("torch.save") as mock_save:
         learner.save_model()
         learner.model_path.parent.mkdir.assert_called_once()
         mock_save.assert_called_once()

def test_load_existing_model(mock_model_cls, mock_optimizer_cls):
    # Simulate existing model file
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.load") as mock_load:
            mock_load.return_value = {
                'model_state_dict': {},
                'optimizer_state_dict': {}
            }
            learner = V2Learner(device="cpu")
            assert learner.is_loaded is True
            learner.model.load_state_dict.assert_called()
