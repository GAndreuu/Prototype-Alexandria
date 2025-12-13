"""
Tests for Action Types
"""
import pytest
from datetime import datetime
from core.agents.action.types import ActionType, ActionStatus, EvidenceType, ActionResult, TestHypothesis

class TestActionTypes:
    def test_action_type_enum(self):
        # Verify core types exist
        assert hasattr(ActionType, 'API_CALL')
        assert hasattr(ActionType, 'PARAMETER_ADJUSTMENT')
        assert hasattr(ActionType, 'DATA_GENERATION')

    def test_action_status_enum(self):
        assert hasattr(ActionStatus, 'PENDING')
        assert hasattr(ActionStatus, 'COMPLETED')
        assert hasattr(ActionStatus, 'FAILED')

    def test_evidence_type_enum(self):
        assert hasattr(EvidenceType, 'SUPPORTING')

class TestActionResult:
    def test_create_result(self):
        result = ActionResult(
            action_id="1",
            action_type=ActionType.API_CALL,
            status=ActionStatus.COMPLETED,
            start_time=datetime.now()
        )
        assert result.action_id == "1"

    def test_result_with_error(self):
        result = ActionResult(
            action_id="1",
            action_type=ActionType.API_CALL,
            status=ActionStatus.FAILED,
            start_time=datetime.now(),
            error_message="Error"
        )
        assert result.error_message == "Error"

class TestTestHypothesis:
    def test_create_hypothesis(self):
        hyp = TestHypothesis(
            hypothesis_id="h1",
            hypothesis_text="test",
            source_cluster=0,
            target_cluster=1,
            test_action=ActionType.API_CALL,
            test_parameters={},
            expected_outcome={},
            validation_criteria={},
            created_at=datetime.now()
        )
        assert hyp.hypothesis_id == "h1"
