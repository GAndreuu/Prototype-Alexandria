"""
Tests for Active Inference Learning
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from core.learning.active_inference import (
    ActiveInferenceAgent, 
    ActiveInferenceConfig, 
    ActionType,  # Has QUERY_SEARCH, EXPLORE_CLUSTER, etc.
    Action, 
    Belief
)

class TestActiveInferenceAgent:
    @pytest.fixture
    def agent(self):
        config = ActiveInferenceConfig()
        return ActiveInferenceAgent(config)

    def test_init(self, agent):
        assert agent is not None

    def test_plan_returns_action(self, agent):
        # plan(observation, horizon, context) - observation is positional
        observation = np.zeros(64)  # state_dim from config
        with patch.object(agent, 'generative_model'):
            actions, values = agent.plan(observation=observation, horizon=3)
            assert isinstance(actions, list)

    def test_detect_knowledge_gaps(self, agent):
        # detect_knowledge_gaps may return dict or list
        gaps = agent.detect_knowledge_gaps()
        assert gaps is not None  # Could be dict or list

class TestActionType:
    def test_action_types_exist(self):
        # ActionType uses auto() enum: QUERY_SEARCH, EXPLORE_CLUSTER, etc.
        assert hasattr(ActionType, 'QUERY_SEARCH')
        assert hasattr(ActionType, 'EXPLORE_CLUSTER')
        assert hasattr(ActionType, 'FILL_GAP')

class TestAction:
    def test_create_action(self):
        action = Action(
            action_type=ActionType.QUERY_SEARCH, 
            target="target", 
            parameters={}, 
            expected_information_gain=0.0, 
            expected_risk=0.0, 
            priority=0.0
        )
        assert action.action_type == ActionType.QUERY_SEARCH

class TestBelief:
    def test_create_belief(self):
        # Belief(concept_id, mean, precision, ...)
        belief = Belief(
            concept_id="test",
            mean=np.zeros(10), 
            precision=np.eye(10)
        )
        assert belief.mean.shape == (10,)

    def test_uncertainty_property(self):
        # uncertainty is a property based on precision
        belief_high = Belief(
            concept_id="high",
            mean=np.zeros(10), 
            precision=np.eye(10) * 0.1  # Low precision = high uncertainty
        )
        belief_low = Belief(
            concept_id="low",
            mean=np.zeros(10), 
            precision=np.eye(10) * 10  # High precision = low uncertainty
        )
        assert belief_high.uncertainty > belief_low.uncertainty

class TestActiveInferenceConfig:
    def test_default_values(self):
        config = ActiveInferenceConfig()
        assert config.learning_rate > 0
        assert config.state_dim == 64
