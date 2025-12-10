"""
Integration tests for Active Inference Shadow Mode in SelfFeedingLoop.

Tests:
1. Loop completes with shadow mode enabled
2. Adapter is called at least once
3. Shadow actions are recorded
4. Disabled shadow mode doesn't call adapter
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from typing import Dict, Any, List

from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig
from core.loop.action_selection import LoopState, AgentAction, ActionType


class MockActiveInferenceAdapter:
    """Mock adapter for testing shadow mode without real AI"""
    
    def __init__(self):
        self.call_count = 0
        self.actions_suggested: List[AgentAction] = []
        self.loop_states_received: List[LoopState] = []
    
    def select_action(self, loop_state: LoopState) -> AgentAction:
        self.call_count += 1
        self.loop_states_received.append(loop_state)
        
        action = AgentAction(
            action_type=ActionType.EXPLORE_CLUSTER,
            target=f"mock_target_{self.call_count}",
            parameters={"cycle": loop_state.cycle, "num_gaps": len(loop_state.gaps)},
            expected_free_energy=0.42,
            information_gain=0.15,
            risk=0.27,
            confidence=0.73,
            source="active_inference"
        )
        self.actions_suggested.append(action)
        return action
    
    def update_after_action(self, action: AgentAction, reward: float) -> None:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "agent_available": True,
            "last_action": self.actions_suggested[-1].to_dict() if self.actions_suggested else None
        }


class TestShadowModeBasics:
    """Basic shadow mode tests"""
    
    def test_loop_completes_with_shadow_mode(self):
        """Test that loop completes successfully with shadow mode enabled"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        # Run single cycle
        cycle = loop.run_cycle()
        
        # Assert loop completed
        assert cycle is not None
        assert cycle.cycle_id >= 0
    
    def test_adapter_called_when_shadow_enabled(self):
        """Test that adapter is called when shadow mode is enabled"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=3,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert adapter.call_count > 0, "ActiveInference adapter was never called"
    
    def test_shadow_actions_recorded(self):
        """Test that shadow actions are recorded in the loop"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert len(loop.shadow_actions) > 0, "No shadow actions recorded"
        assert all(a.source == "active_inference" for a in loop.shadow_actions)
    
    def test_shadow_disabled_no_adapter_calls(self):
        """Test that adapter is NOT called when shadow mode is disabled"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference_shadow=False  # Disabled
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert adapter.call_count == 0, "Adapter should not be called with shadow mode disabled"
        assert len(loop.shadow_actions) == 0


class TestShadowModeIntegration:
    """More detailed integration tests"""
    
    def test_loop_state_passed_correctly(self):
        """Test that LoopState contains correct data"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert len(adapter.loop_states_received) > 0
        state = adapter.loop_states_received[0]
        
        assert isinstance(state, LoopState)
        assert state.cycle == 0  # First cycle
        assert isinstance(state.gaps, list)
        assert isinstance(state.hypotheses, list)
    
    def test_multiple_cycles_accumulate_shadow_actions(self):
        """Test that multiple cycles accumulate shadow actions"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        # Run 3 cycles
        for _ in range(3):
            loop.run_cycle()
        
        assert adapter.call_count == 3
        assert len(loop.shadow_actions) == 3
    
    def test_reset_clears_shadow_actions(self):
        """Test that reset() clears shadow actions"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        assert len(loop.shadow_actions) > 0
        
        loop.reset()
        assert len(loop.shadow_actions) == 0
    
    def test_shadow_action_contains_expected_fields(self):
        """Test that recorded action has all expected fields"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        action = loop.shadow_actions[0]
        assert action.action_type == ActionType.EXPLORE_CLUSTER
        assert "mock_target" in action.target
        assert action.expected_free_energy == 0.42
        assert action.source == "active_inference"


class TestPrimaryMode:
    """Tests for use_active_inference=True (AI as primary decision source)"""
    
    def test_primary_mode_loop_completes(self):
        """Test that loop completes with primary mode enabled"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        cycle = loop.run_cycle()
        
        assert cycle is not None
        assert cycle.cycle_id >= 0
    
    def test_primary_mode_adapter_called(self):
        """Test that adapter is called in primary mode"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert adapter.call_count > 0
    
    def test_primary_mode_ai_actions_tracked(self):
        """Test that AI primary actions are tracked"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        loop.run_cycle()
        
        assert len(loop.ai_primary_actions) > 0
        assert loop.ai_primary_actions[0].source == "active_inference"
    
    def test_primary_mode_action_executed(self):
        """Test that AI-selected action is actually executed"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        cycle = loop.run_cycle()
        
        # At least one action should be executed
        assert cycle.actions_executed >= 1


class MockFailingAdapter:
    """Mock adapter that always fails for testing fallback"""
    
    def __init__(self):
        self.call_count = 0
    
    def select_action(self, loop_state: LoopState) -> AgentAction:
        self.call_count += 1
        raise RuntimeError("Simulated AI failure")
    
    def update_after_action(self, action: AgentAction, reward: float) -> None:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {"call_count": self.call_count}


class TestFallbackBehavior:
    """Tests for fallback to heuristic when AI fails"""
    
    def test_fallback_when_adapter_throws(self):
        """Test that loop continues via heuristic when adapter throws"""
        adapter = MockFailingAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter
        )
        
        # Should not raise, should use fallback
        cycle = loop.run_cycle()
        
        assert cycle is not None
        assert loop.ai_fallback_count == 1
    
    def test_fallback_when_no_adapter(self):
        """Test that loop works without adapter"""
        config = LoopConfig(
            max_hypotheses_per_cycle=2,
            use_active_inference=True  # Enabled but no adapter
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=None  # No adapter
        )
        
        # Should work fine with heuristic
        cycle = loop.run_cycle()
        
        assert cycle is not None


class MockMycelial:
    """Mock MycelialReasoning for testing stats extraction"""
    
    def get_network_stats(self) -> Dict[str, Any]:
        return {
            "active_nodes": 42,
            "active_edges": 128,
            "total_observations": 500,
            "mean_weight": 0.35,
            "max_weight": 0.95,
        }


class MockField:
    """Mock PreStructuralField for testing stats extraction"""
    
    def __init__(self):
        self.manifold = MockManifold()
        self.trigger_count = 17
        self.free_energy = MockFreeEnergy()


class MockManifold:
    def __init__(self):
        self.points = {f"p{i}": None for i in range(10)}  # 10 points
        self.current_dim = 384


class MockFreeEnergy:
    def __init__(self):
        self.last_F = 0.42


class TestRealStats:
    """Tests for real stats in LoopState when components are connected"""
    
    def test_mycelial_stats_populated(self):
        """Test that mycelial stats are extracted when component is present"""
        adapter = MockActiveInferenceAdapter()
        mycelial = MockMycelial()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter,
            mycelial=mycelial
        )
        
        loop.run_cycle()
        
        # Check that adapter received stats
        assert len(adapter.loop_states_received) > 0
        state = adapter.loop_states_received[0]
        assert state.mycelial_stats is not None
        assert len(state.mycelial_stats) > 0
        assert state.mycelial_stats.get("active_nodes") == 42
        assert state.mycelial_stats.get("active_edges") == 128
    
    def test_field_stats_populated(self):
        """Test that field stats are extracted when component is present"""
        adapter = MockActiveInferenceAdapter()
        field = MockField()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter,
            field=field
        )
        
        loop.run_cycle()
        
        # Check that adapter received stats
        assert len(adapter.loop_states_received) > 0
        state = adapter.loop_states_received[0]
        assert state.field_stats is not None
        assert len(state.field_stats) > 0
        assert state.field_stats.get("manifold_points") == 10
        assert state.field_stats.get("trigger_count") == 17
    
    def test_stats_empty_when_no_components(self):
        """Test that stats are empty dicts when components are not connected"""
        adapter = MockActiveInferenceAdapter()
        config = LoopConfig(
            max_hypotheses_per_cycle=1,
            use_active_inference_shadow=True
        )
        
        loop = SelfFeedingLoop(
            config=config,
            active_inference_adapter=adapter,
            mycelial=None,
            field=None
        )
        
        loop.run_cycle()
        
        state = adapter.loop_states_received[0]
        assert state.mycelial_stats == {}
        assert state.field_stats == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
