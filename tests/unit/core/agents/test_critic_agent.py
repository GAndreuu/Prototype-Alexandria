import asyncio
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import json
from core.agents.critic_agent import CriticAgent, RiskLevel, TruthScore, CriticalAssessment

@pytest.fixture
def critic_agent():
    return CriticAgent(gemini_api_key="FIRST4CHARS_MOCK_KEY", risk_tolerance=0.7)

def test_initialization(critic_agent):
    assert critic_agent.gemini_api_key == "FIRST4CHARS_MOCK_KEY"
    assert len(critic_agent.assessment_history) == 0
    assert critic_agent.risk_tolerance == 0.7

async def _async_test_assess_hypothesis(critic_agent):
    """Async implementation"""
    hypothesis = {"id": "hyp1", "description": "Test Hypothesis"}
    supporting = [{"content": "Fact 1", "similarity_score": 0.9}]
    contradicting = []
    
    mock_response_data = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "Analysis. truth score: 0.95. Recommendation: aprovar."
                }]
            }
        }]
    }
    
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        assessment = await critic_agent.assess_hypothesis(hypothesis, supporting, contradicting)
        
        assert assessment.truth_score == 0.95
        assert assessment.recommendation == "aprovar"
        assert len(critic_agent.assessment_history) == 1
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "key=FIRST4CHARS_MOCK_KEY" in args[0]

def test_assess_hypothesis_with_api_mock(critic_agent):
    """Wrapper for async test"""
    asyncio.run(_async_test_assess_hypothesis(critic_agent))

async def _async_test_fallback(critic_agent):
    critic_agent.gemini_api_key = None # Force fallback
    
    hypothesis = {"id": "hyp2", "description": "Matemática e Física são fundamentais"}
    supporting = []
    
    assessment = await critic_agent.assess_hypothesis(hypothesis, supporting)
    
    assert assessment.truth_score > 0.0
    # CriticalAssessment does not have 'reasoning' field, checking coherence score instead
    assert assessment.reasoning_coherence >= 0.0

def test_assess_hypothesis_fallback(critic_agent):
    asyncio.run(_async_test_fallback(critic_agent))

def test_system_feedback(critic_agent):
    """Test feedback generation (async method called via run)"""
    # Manually add history
    assessment = CriticalAssessment(
        hypothesis_id="h1", truth_score=0.9, truth_category=TruthScore.FACTUAL,
        risk_level=RiskLevel.BAIXO, confidence_intervals={}, evidence_quality=0.8,
        contradiction_strength=0.0, reasoning_coherence=0.9, supporting_facts=[],
        contradicting_facts=[], gaps_in_evidence=[], recommendation="aprovar",
        suggested_adjustments=[], assessed_at=datetime.now()
    )
    critic_agent._update_assessment_history(assessment)
    critic_agent._update_system_metrics(assessment)
    
    async def _get_feedback():
        return await critic_agent.get_system_feedback()
        
    feedback = asyncio.run(_get_feedback())
    
    assert feedback.total_assessments == 1
    assert feedback.approval_rate == 1.0
    assert feedback.average_truth_score == 0.9

def test_export_report(critic_agent):
    report_json = critic_agent.export_assessment_report("json")
    data = json.loads(report_json)
    assert "metadata" in data
    assert "system_metrics" in data
