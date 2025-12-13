"""
Tests for core/learning/predictive_coding.py
"""
import pytest
import numpy as np
from core.learning.predictive_coding import PredictiveCodingNetwork, PredictiveCodingConfig

class TestPredictiveCoding:
    """Tests for PredictiveCodingNetwork class."""
    
    @pytest.fixture
    def pc(self):
        """Create predictive coding network instance."""
        config = PredictiveCodingConfig(
            input_dim=64,
            hidden_dims=[32, 16],
            code_dim=8,
            num_iterations=5
        )
        return PredictiveCodingNetwork(config)
    
    def test_init(self, pc):
        """Test initialization."""
        assert pc.config is not None
        assert len(pc.layers) == 3  # 64->32, 32->16, 16->8
    
    def test_infer(self, pc):
        """Test inference."""
        input_data = np.random.randn(64)
        code, stats = pc.infer(input_data, max_iterations=5)
        
        assert code.shape == (8,)
        assert 'final_error' in stats
        
    def test_learn_from_input(self, pc):
        """Test learning."""
        input_data = np.random.randn(64)
        results = pc.learn_from_input(input_data)
        
        assert 'code' in results
        assert 'learning' in results
        assert len(results['learning']) == 3
        
    def test_encode_decode(self, pc):
        """Test encode and decode cycle."""
        input_data = np.random.randn(64)
        
        # Encode
        code = pc.encode(input_data)
        assert code.shape == (8,)
        
        # Decode
        reconstruction = pc.decode(code)
        assert reconstruction.shape == (64,)

    def test_save_load_state(self, pc, tmp_path):
        """Test state persistence."""
        save_path = tmp_path / "pc_network.pkl"
        
        # Initialize with random weights
        pc.infer(np.random.randn(64)) 
        
        pc.save_state(str(save_path))
        
        new_pc = PredictiveCodingNetwork(pc.config)
        success = new_pc.load_state(str(save_path))
        
        assert success
        # Verify if dimensions match
        assert new_pc.layers[0].output_dim == pc.layers[0].output_dim
