"""
Tests for HypothesisExecutor
"""
import pytest
from unittest.mock import Mock, patch
from core.loop.hypothesis_executor import HypothesisExecutor, ActionResult

class TestHypothesisExecutor:
    @pytest.fixture
    def executor(self):
        return HypothesisExecutor()

    def test_init(self, executor):
        assert executor is not None

    def test_hypothesis_to_action(self, executor):
        hypothesis = {
            "hypothesis_text": "Test",
            "confidence_score": 0.9,
            "id": "hyp1"
        }
        action = executor.hypothesis_to_action(hypothesis)
        assert action is not None
        assert action.target is not None

    def test_execute(self, executor):
        hypothesis = {
            "hypothesis_text": "Test",
            "confidence_score": 0.4, 
            "source_cluster": "cluster1"
        }
        
        # Mock _execute_action to avoid real execution logic
        with patch.object(executor, '_execute_action') as mock_run:
             mock_run.return_value = ActionResult(action=Mock(), success=True)
             result = executor.execute(hypothesis)
             assert result.success

    def test_get_stats(self, executor):
        stats = executor.get_stats()
        assert isinstance(stats, dict)
