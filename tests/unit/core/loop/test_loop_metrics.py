"""
Tests for LoopMetrics
"""
import pytest
from core.loop.loop_metrics import LoopMetrics, CycleMetrics

class TestLoopMetrics:
    @pytest.fixture
    def metrics(self):
        return LoopMetrics()

    def test_init(self, metrics):
        assert metrics.total_cycles == 0

    def test_record_cycle(self, metrics):
        # Must start a cycle to get a CycleMetrics object
        cycle = metrics.start_cycle()
        assert isinstance(cycle, CycleMetrics)
        
        cycle.gaps_detected = 1
        cycle.actions_executed = 2
        cycle.actions_successful = 1
        cycle.learning_triggered = True
        
        metrics.record_cycle(cycle)
        
        assert metrics.total_cycles == 1
        assert metrics.total_gaps == 1
        assert metrics.total_successful == 1
        assert metrics.total_learning_events == 1

    def test_get_summary(self, metrics):
        summary = metrics.get_summary()
        assert "total_cycles" in summary
        assert "success_rate" in summary

    def test_is_converged(self, metrics):
        # Default empty is False
        assert metrics.is_converged() is False
        
    def test_save_load(self, metrics, tmp_path):
        f = tmp_path / "metrics.json"
        metrics.save_to_file(str(f))
        assert f.exists()
