"""
Tests for ActionFeedbackCollector
"""
import pytest
from unittest.mock import Mock, patch
from core.loop.feedback_collector import ActionFeedbackCollector, TrainingFeedback

class TestActionFeedbackCollector:
    @pytest.fixture
    def collector(self):
        return ActionFeedbackCollector()

    def test_init(self, collector):
        assert collector.feedback_history == []

    def test_collect_success(self, collector):
        result = {
            "success": True,
            "evidence_found": ["ev1", "ev2"],
            "new_connections": 1,
            "action": {"action_type": "TEST"}
        }
        
        # Mock extract_embeddings inside collect
        # We need to ensure we don't hit the TopologyEngine unless mocked
        with patch.object(collector, '_extract_embeddings', return_value=[Mock(), Mock()]):
             feedback = collector.collect(result)
             assert isinstance(feedback, TrainingFeedback)
             assert feedback.reward_signal > 0
             assert len(feedback.embeddings) == 2

    def test_collect_failure(self, collector):
        result = {
            "success": False,
            "evidence_found": []
        }
        # Failure returns negative reward (min_reward default -0.5)
        feedback = collector.collect(result)
        assert feedback.reward_signal < 0

    def test_get_stats(self, collector):
        stats = collector.get_stats()
        assert stats['total_collected'] == 0
