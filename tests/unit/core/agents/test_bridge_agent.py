
import pytest
import numpy as np
from core.agents.bridge_agent import (
    BridgeAgent, KnowledgeGap, BridgeRequest, BridgeSpec, BridgeCandidate,
    build_bridge_vector, evaluate_bridge_impact, plan_bridge_acquisition
)

@pytest.fixture
def bridge_agent():
    return BridgeAgent()

@pytest.fixture
def mock_gap():
    return KnowledgeGap(
        gap_id="gap1",
        source_concept="Concept A",
        target_concept="Concept B",
        source_vec=np.array([1.0, 0.0, 0.0]),
        target_vec=np.array([0.0, 1.0, 0.0]),
        source_codes=[1, 2],
        target_codes=[3, 4],
        context_tags=["test", "rl"],
        relation_type="missing_mechanism"
    )

def test_build_bridge_vector():
    s = np.array([1.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 0.0])
    v = build_bridge_vector(s, t, w1=0.5, w2=0.5, w3=0.0)
    # 0.5*s + 0.5*t = [0.5, 0.5, 0.0]
    # Norm = sqrt(0.25+0.25) = sqrt(0.5) = 0.707
    # Normalized: ~[0.707, 0.707, 0]
    expected_norm = 1.0
    assert np.isclose(np.linalg.norm(v), expected_norm, atol=1e-5)
    assert v[0] > 0 and v[1] > 0

def test_plan_bridge_acquisition(mock_gap):
    req = plan_bridge_acquisition(mock_gap)
    assert isinstance(req, BridgeRequest)
    assert req.gap_id == "gap1"
    assert req.semantic_query_vec.shape == (3,)
    assert req.bridge_spec.domain == "reinforcement_learning" # inferred from tag 'rl'
    assert "process" in req.text_query or "mechanism" in req.text_query

def test_evaluate_candidate(bridge_agent, mock_gap):
    req = plan_bridge_acquisition(mock_gap)
    
    candidate_vec = np.array([0.0, 0.0, 1.0]) # Orthogonal to source/target
    candidate_meta = {"title": "Test Paper", "doc_id": "d1"}
    memory_vecs = [np.array([1.0, 0.0, 0.0])] # Source is in memory
    
    candidate = bridge_agent.evaluate_candidate(
        mock_gap, req, candidate_vec, candidate_meta, memory_vecs
    )
    
    assert isinstance(candidate, BridgeCandidate)
    assert candidate.title == "Test Paper"
    # Logic checks
    # sim_source = 0, sim_target = 0
    # novelty: 1 - sim([1,0,0], [0,0,1]) = 1 - 0 = 1.0 (Very novel)
    # final_score = alpa*sim_bridge + beta*0 + gamma*1.0
    
    assert candidate.novelty_score == 1.0
    assert candidate.final_score > 0.0

def test_bridge_agent_planning(bridge_agent, mock_gap):
    req = bridge_agent.plan_bridge(mock_gap)
    assert req.gap_id == "gap1"
