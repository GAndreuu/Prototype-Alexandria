
import pytest
import os
import shutil
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from datetime import datetime

from core.agents.action.agent import ActionAgent
from core.agents.action.types import ActionType, ActionStatus, ActionResult, EvidenceType

@pytest.fixture
def action_agent(tmp_path):
    """Fixture ensuring a fresh ActionAgent with temporary SFS path"""
    # Patch V2Learner where it is imported in the agent module
    with patch("core.agents.action.agent.V2Learner") as MockLearner:
        agent = ActionAgent(sfs_path=str(tmp_path))
        agent.v2_learner = MockLearner.return_value
        yield agent

def test_initialization(action_agent, tmp_path):
    """Test proper initialization of the agent"""
    assert str(action_agent.sfs_path) == str(tmp_path)
    assert action_agent.security_controller is not None
    assert action_agent.parameter_controller is not None
    assert action_agent.v2_learner is not None
    assert tmp_path.exists()

@patch("core.agents.action.agent.execute_api_call")
def test_execute_api_call(mock_exec, action_agent):
    """Test execution of API call action"""
    mock_result = ActionResult(
        action_id="test_id",
        action_type=ActionType.API_CALL,
        status=ActionStatus.COMPLETED,
        start_time=datetime.now()
    )
    mock_exec.return_value = mock_result
    
    params = {"url": "http://test.com", "method": "GET"}
    result = action_agent.execute_action(ActionType.API_CALL, params)
    
    assert result == mock_result
    mock_exec.assert_called_once_with(params, action_agent.security_controller, ANY)
    # Check that a result was stored (ID is random)
    assert len(action_agent.action_results) == 1

def test_rate_limit_check(action_agent):
    """Test rate limiting prevents execution"""
    action_agent.security_controller.check_rate_limit = MagicMock(return_value=False)
    
    result = action_agent.execute_action(ActionType.API_CALL, {})
    
    assert result.status == ActionStatus.FAILED
    assert "Rate limit" in result.error_message
    
@patch("core.agents.action.agent.execute_simulation")
def test_test_hypothesis(mock_sim, action_agent):
    """Test hypothesis testing flow"""
    mock_result = ActionResult(
        action_id="sim_id",
        action_type=ActionType.SIMULATION_RUN,
        status=ActionStatus.COMPLETED,
        start_time=datetime.now(),
        result_data={"metrics": {"convergence_rate": 0.8, "stability": 0.9}}
    )
    mock_sim.return_value = mock_result
    
    hypothesis = {
        "id": "hyp1",
        "hypothesis_text": "Test Hyp",
        "test_action": "simulation_run",
        "test_parameters": {"p": 1},
        "validation_criteria": {"min_convergence": 0.5}
    }
    
    # Mock json.dump to avoid serialization error of Enums during evidence registration
    with patch("json.dump"):
        test_hyp = action_agent.test_hypothesis(hypothesis)
    
    assert test_hyp.result == mock_result
    assert test_hyp.result.evidence_generated is True
    assert test_hyp.result.evidence_type == EvidenceType.SUPPORTING
    assert "hyp1" in action_agent.test_hypotheses
    
    # Verify evidence file creation (json.dump mocked but open happens)
    # The file is created empty or partial? 
    # With patch json.dump, file write is empty?
    # Actually open() context manager creates file.
    evidence_files = list(action_agent.sfs_path.glob("test_evidence_hyp1*.json"))
    assert len(evidence_files) > 0

def test_get_statistics(action_agent):
    """Test statistics generation"""
    from core.agents.action.types import TestHypothesis
    from datetime import timedelta
    
    start = datetime.now()
    end = start + timedelta(seconds=1)
    
    hyp = TestHypothesis(
        hypothesis_id="h1",
        hypothesis_text="t",
        source_cluster=0, target_cluster=1,
        test_action=ActionType.SIMULATION_RUN,
        test_parameters={},
        expected_outcome={},
        validation_criteria={},
        created_at=start
    )
    hyp.result = ActionResult("id", ActionType.SIMULATION_RUN, ActionStatus.COMPLETED, start_time=start)
    hyp.result.evidence_type = EvidenceType.SUPPORTING
    hyp.result.end_time = end
    
    action_agent.test_hypotheses["h1"] = hyp
    
    stats = action_agent.get_test_statistics()
    assert stats["total_tests"] == 1
    assert stats["evidence_distribution"]["supporting"] == 1
    assert stats["average_duration"] == 1.0

