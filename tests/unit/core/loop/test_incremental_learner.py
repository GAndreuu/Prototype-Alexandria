"""
Tests for IncrementalLearner
"""
import pytest
from unittest.mock import Mock
import numpy as np
from core.loop.incremental_learner import IncrementalLearner

class TestIncrementalLearner:
    @pytest.fixture
    def learner(self):
        return IncrementalLearner(batch_threshold=2)

    def test_init(self, learner):
        assert learner.batch_threshold == 2

    def test_add_feedback(self, learner):
        feedback = {
            "embeddings": [np.random.randn(384)],
            "reward_signal": 0.5,
            "should_learn": True
        }
        # First add (below threshold)
        triggered = learner.add_feedback(feedback)
        assert not triggered
        assert len(learner.current_batch) == 1
        
        # Second add (hits threshold)
        triggered = learner.add_feedback(feedback)
        assert triggered
        # Should be cleared after learning (simulated since no v2_learner provided)
        assert len(learner.current_batch) == 0

    def test_force_learn(self, learner):
        learner.add_feedback({"embeddings": [np.random.randn(384)], "reward_signal": 0.5, "should_learn": True})
        metrics = learner.force_learn()
        assert metrics is not None
        assert len(learner.current_batch) == 0

    def test_get_stats(self, learner):
        stats = learner.get_stats()
        assert "total_learned" in stats
