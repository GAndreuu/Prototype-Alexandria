import pytest
from unittest.mock import MagicMock, patch
from core.integrations.alexandria_unified import AlexandriaCore

# =============================================================================
# TESTS: AlexandriaUnified (Pure Mock)
# =============================================================================

@pytest.fixture
def mock_dependencies():
    """Mock all internal dependencies of AlexandriaCore."""
    # We must patch the classes WHERE THEY ARE IMPORTED in the target module
    with patch("core.integrations.alexandria_unified.NemesisBridgeIntegration") as MockNemesis, \
         patch("core.integrations.alexandria_unified.LearningFieldIntegration") as MockLearning, \
         patch("core.integrations.alexandria_unified.AbductionCompositionalIntegration") as MockAbduction, \
         patch("core.integrations.alexandria_unified.AgentsCompositionalIntegration") as MockAgents, \
         patch("core.integrations.alexandria_unified.LoopCompositionalIntegration") as MockLoop, \
         patch("core.integrations.alexandria_unified.VQVAEManifoldBridge") as MockBridge:
        
        yield {
            "nemesis": MockNemesis,
            "learning": MockLearning,
            "abduction": MockAbduction,
            "agents": MockAgents,
            "loop": MockLoop,
            "bridge": MockBridge
        }

@pytest.fixture
def alexandria(mock_dependencies):
    """AlexandriaCore instance with mocked subs."""
    # Instantiating with bridge=None to force internal bridge creation logic or not
    # Ideally we pass our mocks or rely on the patches during init
    
    # We simulate that the factory method is used or just Init
    # If we use direct init, we might need to pass dependencies if they are optional arguments
    # But AlexandriaCore.__init__ takes bridge/compositional.
    # _init_integrations instantiates sub-integrations using the bridge.
    
    # Let's assume we pass a Mock bridge so _init_integrations runs
    mock_bridge = MagicMock()
    core = AlexandriaCore(bridge=mock_bridge)
    return core

def test_initialization(alexandria, mock_dependencies):
    """Test if all subsystems are initialized."""
    # _init_integrations is called in __init__
    
    # Verify sub-integrations were instantiated
    # Note: AlexandriaCore only inits them if imports were successful (which mocks simulate)
    # AND if self.bridge is not None (which we passed)
    
    mock_dependencies["nemesis"].assert_called()
    mock_dependencies["learning"].assert_called()
    assert alexandria.nemesis is not None
    assert alexandria.learning is not None

def test_cognitive_cycle(alexandria):
    """Test the full cognitive cycle flow."""
    # Setup mocks for cycle components
    
    # 1. Perception (Learning)
    alexandria.learning.process_observation.return_value = {
        'code': 123, 'prediction': {}, 'free_energy': 0.1
    }
    
    # 2. Reasoning (Abduction)
    # detect_gaps_geometric returns list of gaps
    mock_gap = MagicMock() 
    mock_gap.gap_id = "g1"
    mock_gap.gap_type = "novelty"
    mock_gap.priority_score = 0.9
    alexandria.abduction.detect_gaps_geometric.return_value = [mock_gap]
    
    mock_hyp = MagicMock()
    mock_hyp.hypothesis_id = "h1"
    mock_hyp.confidence_score = 0.8
    alexandria.abduction.generate_geodesic_hypotheses.return_value = [mock_hyp]
    
    # 3. Action (Agents or Nemesis)
    # Let's say Nemesis acts
    mock_action_result = MagicMock()
    mock_action_result.action_type = "explore"
    mock_action_result.target = "node_x"
    mock_action_result.geometric_efe.total = 0.5
    alexandria.nemesis.select_action_geometric.return_value = mock_action_result
    
    # Execute
    observation = MagicMock()
    result = alexandria.cognitive_cycle(observation)
    
    # Assert
    assert result.iteration == 1
    assert result.free_energy == 0.1
    assert result.action_selected['type'] == 'explore'
    
    # Verify calls
    alexandria.learning.process_observation.assert_called_with(observation, None)
    alexandria.abduction.detect_gaps_geometric.assert_called()
    alexandria.nemesis.select_action_geometric.assert_called()

def test_autonomous_run(alexandria):
    """Test autonomous run delegation."""
    # Setup Loop mock
    alexandria.loop.autonomous_cycle.return_value = {'status': 'success'}
    
    # Execute
    res = alexandria.autonomous_run(MagicMock())
    
    # Assert
    assert res['status'] == 'success'
    alexandria.loop.autonomous_cycle.assert_called()
