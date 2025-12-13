"""
Tests for SecurityController
"""
import pytest
from unittest.mock import Mock, patch
from core.agents.action.security_controller import SecurityController
from core.agents.action.types import ActionType, ActionResult, ActionStatus

class TestSecurityController:
    @pytest.fixture
    def controller(self):
        return SecurityController()

    def test_init(self, controller):
        assert controller.allowed_apis is not None
        assert isinstance(controller.allowed_apis, list)

    def test_check_rate_limit(self, controller):
        # Mock time to avoid flaky tests
        with patch('time.time', return_value=1000):
            # check_rate_limit signature: (action_type, user_id="system")
            assert controller.check_rate_limit(ActionType.API_CALL)
            
    def test_log_action(self, controller):
        action = Mock(spec=ActionResult)
        action.action_id = "1"
        action.action_type = ActionType.API_CALL
        action.status = ActionStatus.COMPLETED
        # Need to mock .value access if ActionType is Enum, but here we used real ActionType
        
        controller.log_action(action, {"detail": "test"})
        logs = controller.get_audit_log()
        assert len(logs) == 1
        assert logs[0]['action_id'] == "1"
