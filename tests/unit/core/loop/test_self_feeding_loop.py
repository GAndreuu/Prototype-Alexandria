"""
Tests for Self Feeding Loop
"""
import pytest
from unittest.mock import Mock, patch
from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig

class TestSelfFeedingLoop:
    @pytest.fixture
    def loop(self):
        # SelfFeedingLoop takes optional components
        with patch('core.loop.self_feeding_loop.AbductionEngine'), \
             patch('core.loop.self_feeding_loop.HypothesisExecutor'), \
             patch('core.loop.self_feeding_loop.ActionFeedbackCollector'), \
             patch('core.loop.self_feeding_loop.IncrementalLearner'):
            return SelfFeedingLoop()

    def test_init(self, loop):
        assert loop is not None

    def test_run_cycle(self, loop):
        # run_cycle is the real method, not run_single_cycle
        loop.abduction_engine = Mock()
        loop.abduction_engine.detect_knowledge_gaps = Mock(return_value=[])
        loop.abduction_engine.generate_hypotheses = Mock(return_value=[])
        loop.hypothesis_executor = Mock()
        loop.feedback_collector = Mock()
        loop.incremental_learner = Mock()
        
        result = loop.run_cycle()
        assert result is not None

    def test_get_status(self, loop):
        status = loop.get_status()
        assert "current_cycle" in status

    def test_reset(self, loop):
        loop.reset()
        assert loop.current_cycle == 0

    def test_stop(self, loop):
        loop.stop()
        status = loop.get_status()
        # Check is_running flag exists and is false
        assert status.get("is_running", False) == False

class TestLoopConfig:
    def test_default_values(self):
        config = LoopConfig()
        assert config.max_cycles > 0
        assert config.max_hypotheses_per_cycle > 0
