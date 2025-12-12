
import pytest
from unittest.mock import MagicMock
from core.loop.hypothesis_executor import HypothesisExecutor, ExecutableAction, ExecutionActionType
from core.reasoning.mycelial_reasoning import MycelialReasoning

@pytest.fixture
def mycelial():
    mr = MycelialReasoning()
    mr.reset()
    # Add some initial data
    # Create a small graph: (0,1) <-> (0,2)
    mr.connect_nodes((0, 1), (0, 2), weight_delta=0.5)
    return mr

@pytest.fixture
def executor(mycelial):
    return HypothesisExecutor(mycelial_reasoning=mycelial)

def test_bridge_concepts_real(executor, mycelial):
    """Test that BRIDGE_CONCEPTS actually updates the graph."""
    
    # Check initial state (should be 0 or low)
    # connection between (0,1) and (1,5) does not exist
    
    action = ExecutableAction(
        action_type=ExecutionActionType.BRIDGE_CONCEPTS,
        target="Test Bridge",
        parameters={
            "source": "Concept A", 
            "target": "Concept B",
            "source_node": (0, 1),
            "target_node": (1, 5)
        }
    )
    
    result = executor._execute_action(action)
    
    assert result.success is True
    assert "Mycelial Connection created" in result.evidence_found[0]
    
    # Verify graph state
    neighbors = mycelial.graph[(0, 1)]
    assert (1, 5) in neighbors
    assert neighbors[(1, 5)] >= 0.1

def test_explore_cluster_real(executor, mycelial):
    """Test that EXPLORE_CLUSTER retrieves neighbors."""
    
    # Initial graph has (0,1)<->(0,2) with w=0.5
    
    action = ExecutableAction(
        action_type=ExecutionActionType.EXPLORE_CLUSTER,
        target="Explore Node (0,1)",
        parameters={
            "node_id": (0, 1)
        }
    )
    
    result = executor._execute_action(action)
    
    assert result.success is True
    # Should find (0,2)
    ev_str = str(result.evidence_found)
    assert "(0, 2)" in ev_str 
    assert "w=0.5" in ev_str

def test_explore_cluster_empty(executor):
    """Test exploring a non-existent node."""
    action = ExecutableAction(
        action_type=ExecutionActionType.EXPLORE_CLUSTER,
        target="Explore Void",
        parameters={
            "node_id": (99, 99)
        }
    )
    
    result = executor._execute_action(action)
    # Should be successful but empty or just low evidence?
    # Logic: success = len(evidence) > 0.
    # If no neighbors, evidence is empty -> success=False.
    
    assert result.success is False
    assert len(result.evidence_found) == 0

def test_bridge_missing_params(executor, mycelial):
    """Test BRIDGE_CONCEPTS without node_ids (fallback to logical)."""
    action = ExecutableAction(
        action_type=ExecutionActionType.BRIDGE_CONCEPTS,
        target="Test Logical",
        parameters={
            "source": "Concept A",
            "target": "Concept B"
            # Missing *_node params
        }
    )
    
    result = executor._execute_action(action)
    
    # Should succeed logically
    assert result.success is True
    assert "Logical Connection created" in result.evidence_found[0]
    
    # Graph should NOT change
    neighbors = mycelial.graph.get((99,1), {})
    assert len(neighbors) == 0
