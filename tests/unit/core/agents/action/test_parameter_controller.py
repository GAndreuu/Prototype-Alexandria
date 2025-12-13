"""
Tests for ParameterController
"""
import pytest
import os
from unittest.mock import patch
from core.agents.action.parameter_controller import ParameterController

class TestParameterController:
    @pytest.fixture
    def controller(self):
        return ParameterController()

    def test_init(self, controller):
        assert controller.parameter_history == []
        assert "V11_BETA" in controller.supported_parameters

    def test_get_parameter(self, controller):
        # Default current is 1.0 from source code
        assert controller.get_parameter("V11_BETA") == 1.0

    def test_adjust_parameter(self, controller):
        success = controller.adjust_parameter("V11_BETA", 2.0)
        assert success
        assert controller.get_parameter("V11_BETA") == 2.0
        
    def test_adjust_parameter_invalid_name(self, controller):
        success = controller.adjust_parameter("INVALID_PARAM", 1.0)
        assert not success

    def test_adjust_parameter_out_of_range(self, controller):
        # Min is 0.1
        success = controller.adjust_parameter("V11_BETA", 0.05)
        assert not success

    def test_reset_parameter_no_default(self, controller):
        # Some params in source don't have 'default' key, only 'current'
        # reset should fail safely
        res = controller.reset_parameter("V11_BETA")
        # It's false because 'default' key is missing in the dict in source
        assert res is False

    def test_get_history(self, controller):
        controller.adjust_parameter("V11_BETA", 2.0)
        history = controller.get_parameter_history("V11_BETA")
        assert len(history) == 1
        assert history[0]["new_value"] == 2.0
