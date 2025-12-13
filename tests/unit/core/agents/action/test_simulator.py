"""
Tests for core/agents/action/test_simulator.py (The TestSimulator Class)
"""
import pytest
from unittest.mock import Mock, patch
from core.agents.action.test_simulator import TestSimulator
from core.agents.action.types import ActionType, ActionStatus

class TestTestSimulatorClass:
    """Tests for the TestSimulator class (defined in core/agents/action/test_simulator.py)."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator with mocked agent."""
        mock_agent = Mock()
        mock_agent.parameter_controller = Mock()
        mock_agent.parameter_controller.get_parameter.return_value = 0.5
        
        # Mock execute_action responses for the simulation loop
        # The loop runs multiple times, so we need side_effect to return mocks
        def side_effect(*args, **kwargs):
            action_type = kwargs.get('action_type')
            if action_type == ActionType.PARAMETER_ADJUSTMENT:
                return Mock(status=ActionStatus.COMPLETED)
            elif action_type == ActionType.SIMULATION_RUN:
                return Mock(
                    status=ActionStatus.COMPLETED, 
                    result_data={"metrics": {"convergence_rate": 0.5, "stability": 0.6}},
                    duration=1.0
                )
            return Mock(status=ActionStatus.FAILED)
            
        mock_agent.execute_action.side_effect = side_effect
        
        return TestSimulator(action_agent=mock_agent)

    def test_init(self, simulator):
        """Test initialization."""
        assert simulator is not None
        assert simulator.action_agent is not None

    def test_simulate_v11_parameter_test(self, simulator):
        """Test parameter simulation."""
        hypothesis = {
            "id": "hyp1",
            "test_parameters": {
                "parameter": "V11_BETA",
                "values": [0.5]
            }
        }
        
        result = simulator.simulate_v11_parameter_test(hypothesis)
        
        assert result['simulation_name'] == 'V11_BETA_optimization'
        assert result['best_value'] == 0.5
        
    def test_get_simulation_report(self, simulator):
        """Test report generation."""
        # Add some dummy history
        simulator.simulation_history = [{"best_accuracy": 0.8, "simulation_name": "test"}]
        report = simulator.get_simulation_report()
        assert report['total_simulations'] == 1
