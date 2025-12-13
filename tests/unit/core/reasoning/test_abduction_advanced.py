"""
Tests for Abduction Engine Advanced
"""
import pytest
from unittest.mock import Mock, patch
from core.reasoning.abduction_engine import AbductionEngine

class TestAbductionEngineAdvanced:
    @pytest.fixture
    def engine(self):
        # Avoid SFS loading
        with patch('core.reasoning.abduction_engine.AbductionEngine._load_cluster_labels'), \
             patch('builtins.open'):
            return AbductionEngine(sfs_path="test.jsonl")

    def test_generate_hypotheses(self, engine):
        with patch.object(engine, 'detect_knowledge_gaps') as mock_detect:
             mock_detect.return_value = []
             hyps = engine.generate_hypotheses()
             assert isinstance(hyps, list)
