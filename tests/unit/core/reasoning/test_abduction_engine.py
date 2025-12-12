
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from core.reasoning.abduction_engine import (
    AbductionEngine, Hypothesis, KnowledgeGap, ValidationTest
)

# Mocks for dependencies
# Mock classes
class MockGraph(dict):
    def has_edge(self, u, v):
        return v in self.get(u, {})

@pytest.fixture
def mock_causal_engine():
    engine = MagicMock()
    # Mock causal graph using custom class to allow method attachment
    graph = MockGraph({
        "cluster_A": {"cluster_B": 0.8},
        "cluster_B": {"cluster_C": 0.9},
        "cluster_D": {} # Orphaned
    })
    engine.causal_graph = graph
    
    # Mock memory retrieval
    engine.memory.retrieve.return_value = [
        {"content": "Doc about A and B", "metadata": {"created_at": "2023-01-01"}},
        {"content": "Another doc", "metadata": {"created_at": "2023-01-02"}}
    ]
    
    return engine

@pytest.fixture
def mock_topology():
    topo = MagicMock()
    topo.encode.return_value = np.random.randn(1, 384) # Mock embedding
    return topo

@pytest.fixture
def abduction_engine(mock_causal_engine, mock_topology):
    # Patch internal imports in __init__
    with patch.dict('sys.modules', {
        'core.topology.topology_engine': MagicMock(),
        'core.memory.semantic_memory': MagicMock(),
        'core.reasoning.causal_reasoning': MagicMock(),
        # 'core.reasoning.mycelial_reasoning': MagicMock() # Patched in specific test if needed
    }):
        # Initialize
        engine = AbductionEngine(fast_mode=True)
        # Inject mocks
        engine.causal_engine = mock_causal_engine
        engine.topology = mock_topology
        return engine

def test_initialization(abduction_engine):
    assert abduction_engine.fast_mode is True
    assert abduction_engine.causal_engine is not None
    assert len(abduction_engine.hypotheses) == 0

def test_detect_knowledge_gaps(abduction_engine, mock_causal_engine):
    # Setup graph with an orphaned cluster D (defined in fixture)
    
    gaps = abduction_engine.detect_knowledge_gaps(min_orphaned_score=0.1)
    
    # Check if orphaned cluster D was detected
    orphaned_gaps = [g for g in gaps if g.gap_type == "orphaned_cluster"]
    assert len(orphaned_gaps) >= 1
    
    # Check if ANY of the detected orphaned gaps refers to D
    detected_clusters = []
    for g in orphaned_gaps:
        detected_clusters.extend(g.affected_clusters)
        
    assert "cluster_D" in detected_clusters or any("cluster_D" in g.description for g in orphaned_gaps)

    # Check for missing connections (A and D might be similar if mocked?)
    # We need to mock _calculate_semantic_similarity for deterministic checks
    with patch.object(abduction_engine, '_calculate_semantic_similarity', return_value=0.8):
        # Force high similarity between A and D
        gaps_sim = abduction_engine._find_missing_connections()
        assert len(gaps_sim) > 0

def test_generate_hypotheses(abduction_engine):
    # Mock gap detection to return a specific gap
    gap = KnowledgeGap(
        gap_id="gap1",
        gap_type="orphaned_cluster",
        description="Test orphaned",
        affected_clusters=["cluster_D"],
        priority_score=0.9,
        candidate_hypotheses=[],
        detected_at=datetime.now()
    )
    abduction_engine.knowledge_gaps = {"gap1": gap}
    
    # Mock related clusters finding
    with patch.object(abduction_engine, '_find_related_clusters', return_value=["cluster_A"]):
        hypotheses = abduction_engine.generate_hypotheses(max_hypotheses=5)
        
        assert len(hypotheses) > 0
        assert hypotheses[0].source_cluster.startswith("cluster_D")
        assert hypotheses[0].target_cluster.startswith("cluster_A")

def test_evidence_calculation_fast(abduction_engine):
    # Test _calculate_evidence_fast
    # Graph has A->B (0.8)
    score = abduction_engine._calculate_evidence_fast("cluster_A", "cluster_B")
    assert score >= 0.5 # Should be boosted by graph connection
    
    # Disconnected pairs
    score_disc = abduction_engine._calculate_evidence_fast("cluster_A", "cluster_D")
    # Depends on semantic similarity, which we haven't mocked here yet, 
    # but graph evidence is 0. Base is 0.3.
    assert score_disc >= 0.2 # Standard range

def test_consolidate_knowledge(abduction_engine, mock_topology):
    hypothesis = {
        'source': 'Concept A',
        'target': 'Concept B',
        'relation': 'causes',
        'validation_score': 0.8
    }
    
    # Patch the class in its ORIGINAL module
    with patch('core.reasoning.mycelial_reasoning.MycelialVQVAE') as MockVQVAE:
        mock_wrapper = MagicMock()
        mock_wrapper.encode.return_value = MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(flatten=lambda: np.array([1, 2]))))
        MockVQVAE.load_default.return_value = mock_wrapper
             
        # Allow import in abduction_engine to find this mocked class
        with patch.dict('sys.modules', {'core.reasoning.mycelial_reasoning': MagicMock(MycelialVQVAE=MockVQVAE)}):
             success = abduction_engine.consolidate_knowledge(hypothesis)
             assert success is True
             mock_wrapper.mycelial.observe_sequence.assert_called()

def test_check_temporal_sequence(abduction_engine, mock_causal_engine):
    # Setup docs with timestamps
    doc1 = {"metadata": {"created_at": "2023-01-01T10:00:00"}}
    doc2 = {"metadata": {"created_at": "2023-01-02T10:00:00"}}
    
    mock_causal_engine.memory.retrieve.side_effect = [
        [doc1], # Source docs (older)
        [doc2]  # Target docs (newer)
    ]
    
    score = abduction_engine._check_temporal_sequence("old", "new")
    assert score == 0.8 # Target is newer -> causality plausible

    mock_causal_engine.memory.retrieve.side_effect = [
        [doc2], # Source docs (newer)
        [doc1]  # Target docs (older)
    ]
    score_inverted = abduction_engine._check_temporal_sequence("new", "old")
    assert score_inverted == 0.2 # Inverted causality
