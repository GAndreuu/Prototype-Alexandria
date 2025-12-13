"""
Tests for core/learning/integration_layer.py
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from core.learning.integration_layer import AlexandriaIntegratedPipeline, IntegrationConfig

class TestIntegrationLayer:
    """Tests for AlexandriaIntegratedPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create integration pipeline with mocked modules."""
        # Use create=True to handle cases where optional imports failed
        with patch('core.learning.integration_layer.HAS_META_HEBBIAN', True), \
             patch('core.learning.integration_layer.HAS_ACTIVE_INFERENCE', True), \
             patch('core.learning.integration_layer.create_meta_hebbian_system', create=True) as MockMH, \
             patch('core.learning.integration_layer.create_active_inference_system', create=True) as MockAI, \
             patch('core.learning.integration_layer.IsomorphicPredictiveCoding') as MockPC:
            
            # Setup Mock Returns
            mock_pc_instance = MockPC.return_value
            mock_pc_instance.process.return_value = (np.zeros(384), {'error': 0.1})
            mock_pc_instance.total_processed = 10
            mock_pc_instance.total_surprise = 0.5
            
            mock_ai_instance = MockAI.return_value
            mock_ai_instance.perception_action_cycle.return_value = {
                'action_taken': 'explore', 
                'gaps_detected': []
            }
            
            config = IntegrationConfig()
            pipeline = AlexandriaIntegratedPipeline(config=config)
            
            # Setup sparse_adapter attributes if it was created
            if pipeline.sparse_adapter:
                pipeline.sparse_adapter.update_history = [{'updates': 1, 'mean_delta': 0.1}]
            
            return pipeline

    def test_init(self, pipeline):
        """Test initialization."""
        assert pipeline is not None
        assert pipeline.pc is not None
        # Should be not None if HAS_META_HEBBIAN was patched to True
        assert pipeline.meta_hebbian is not None

    def test_process_text(self, pipeline):
        """Test full pipeline processing."""
        pipeline._get_embedding = Mock(return_value=np.zeros(384))
        pipeline._quantize = Mock(return_value=np.array([0, 1]))
        pipeline._observe_mycelial = Mock(return_value={'observed': True})
        
        result = pipeline.process_text("test input")
        
        assert 'stages' in result
        assert 'surprise' in result
        pipeline.pc.process.assert_called()
        
    def test_process_embedding(self, pipeline):
        """Test embedding processing."""
        pipeline._quantize = Mock(return_value=np.array([0, 1]))
        pipeline._observe_mycelial = Mock(return_value={'observed': True})
        
        emb = np.zeros(384)
        result = pipeline.process_embedding(emb)
        
        assert result['stages']['predictive_coding'] is not None

    def test_get_system_status(self, pipeline):
        """Test status retrieval."""
        status = pipeline.get_system_status()
        assert 'modules' in status
        assert status['modules']['predictive_coding'] is True
