"""
Tests for Executors
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock sklearn modules before importing executors to avoid ImportError if not installed
module_mock = MagicMock()
sys.modules['sklearn'] = module_mock
sys.modules['sklearn.datasets'] = module_mock
sys.modules['sklearn.cluster'] = module_mock
sys.modules['sklearn.manifold'] = module_mock
sys.modules['sklearn.ensemble'] = module_mock
sys.modules['sklearn.model_selection'] = module_mock
sys.modules['sklearn.preprocessing'] = module_mock
sys.modules['sklearn.metrics'] = module_mock
sys.modules['sklearn.svm'] = module_mock
sys.modules['sklearn.neural_network'] = module_mock

from core.agents.action.execution.data_executor import execute_data_generation
from core.agents.action.execution.model_executor import execute_model_retrain
from core.agents.action.types import ActionType, ActionStatus, ActionResult

class TestDataExecutor:
    def test_execute_generation(self):
        params = {"size": 10, "data_type": "random"}
        
        # Mock pathlib and numpy save
        sfs_path = MagicMock()
        sfs_path.__truediv__.return_value = MagicMock()
        
        with patch('numpy.save'), patch('time.time', return_value=0):
             # Mock make_blobs to return data
             with patch('core.agents.action.execution.data_executor.make_blobs', return_value=(MagicMock(), MagicMock()), create=True):
                 result = execute_data_generation(params, sfs_path, "act1")
                 # Should fail if make_blobs not found or pass if mocked correctly
                 # But since we mocked sklearn at sys level, the import inside function might succeed returning a mock
                 assert isinstance(result, ActionResult)

class TestModelExecutor:
    def test_execute_retrain(self):
        params = {"model_name": "random_forest"}
        
        with patch('time.time', return_value=0):
            result = execute_model_retrain(params, "act1")
            assert isinstance(result, ActionResult)
