"""
Tests for core/agents/action/evidence_registrar.py
"""
import pytest
from unittest.mock import Mock, patch, mock_open
from core.agents.action.evidence_registrar import EvidenceRegistrar
from core.agents.action.types import ActionResult, ActionStatus, EvidenceType
from datetime import datetime

class TestEvidenceRegistrar:
    """Tests for EvidenceRegistrar class."""
    
    @pytest.fixture
    def registrar(self):
        """Create registrar with mocked dependencies."""
        mock_agent = Mock()
        mock_agent.sfs_path = Mock()
        # Mock path construction (agent.sfs_path / filename)
        mock_agent.sfs_path.__truediv__ = Mock(return_value="mock_path.md")
        
        mock_sfs = Mock()
        mock_sfs.index_file.return_value = 5 # chunks indexed
        
        return EvidenceRegistrar(action_agent=mock_agent, sfs_instance=mock_sfs)

    def test_init(self, registrar):
        """Test initialization."""
        assert registrar is not None
        assert registrar.sfs is not None

    def test_register_action_evidence(self, registrar):
        """Test registering action evidence."""
        result = Mock(spec=ActionResult)
        result.action_id = "act1"
        result.action_type = Mock()
        result.action_type.value = "test_action"
        result.status = ActionStatus.COMPLETED
        result.duration = 1.0
        result.result_data = {}
        result.start_time = datetime.now()
        result.evidence_type = EvidenceType.SUPPORTING
        
        with patch("builtins.open", mock_open()):
             ev_id = registrar.register_action_evidence(result)
             assert ev_id.startswith("ACTION_EVID_")
             registrar.sfs.index_file.assert_called()

    def test_register_simulation_evidence(self, registrar):
        """Test registering simulation evidence."""
        sim_data = {"simulation_name": "test"}
        with patch("builtins.open", mock_open()), patch("time.time", return_value=123):
             ev_id = registrar.register_simulation_evidence(sim_data)
             assert ev_id.startswith("SIM_EVID_")
             
    def test_get_evidence_statistics(self, registrar):
        """Test stats."""
        stats = registrar.get_evidence_statistics()
        assert stats['total_evidence'] == 0
