"""
Tests for Learning Field Integrations (Geometric PC & AI)
"""
import pytest
from unittest.mock import Mock
import numpy as np
from core.integrations.learning_field_integration import GeometricPredictiveCoding, GeometricActiveInference

class TestGeometricPredictiveCoding:
    @pytest.fixture
    def gpc(self):
        bridge = Mock()
        return GeometricPredictiveCoding(bridge)
        
    def test_init(self, gpc):
        assert gpc is not None

class TestGeometricActiveInference:
    @pytest.fixture
    def gai(self):
        bridge = Mock()
        return GeometricActiveInference(bridge)
        
    def test_init(self, gai):
        assert gai is not None
