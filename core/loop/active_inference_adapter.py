"""
Active Inference Adapter
========================

Wraps ActiveInferenceAgent to implement ActionSelectionAdapter protocol
for integration with SelfFeedingLoop.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

from .action_selection import ActionSelectionAdapter, AgentAction, LoopState, ActionType

logger = logging.getLogger(__name__)


# Map ActiveInference ActionType names to our unified ActionType
_ACTION_TYPE_MAP = {
    "QUERY_SEARCH": ActionType.QUERY_SEARCH,
    "EXPLORE_CLUSTER": ActionType.EXPLORE_CLUSTER,
    "DEEPEN_TOPIC": ActionType.DEEPEN_TOPIC,
    "BRIDGE_CONCEPTS": ActionType.BRIDGE_CONCEPTS,
    "FILL_GAP": ActionType.FILL_GAP,
    "CONSOLIDATE": ActionType.CONSOLIDATE,
    "FOLLOW_CONNECTION": ActionType.FOLLOW_CONNECTION,
    "REST": ActionType.REST,
}


class ActiveInferenceActionAdapter:
    """
    Adapter that connects ActiveInferenceAgent to SelfFeedingLoop.
    
    Responsibilities:
    - Convert LoopState → ActiveInference observations/beliefs
    - Convert ActiveInference Action → AgentAction
    - Track call statistics
    """
    
    def __init__(self, topology_engine=None, state_dim: int = 64):
        self.topology_engine = topology_engine
        self.state_dim = state_dim
        self.agent = None
        self._init_agent()
        
        # Stats
        self.call_count = 0
        self.last_action: Optional[AgentAction] = None
        self.action_history: List[AgentAction] = []
    
    def _init_agent(self):
        """Initialize ActiveInferenceAgent"""
        try:
            from core.learning.active_inference import (
                ActiveInferenceAgent,
                ActiveInferenceConfig
            )
            config = ActiveInferenceConfig(
                state_dim=self.state_dim,
                planning_horizon=3,
                num_action_samples=10,
                temperature=1.0
            )
            self.agent = ActiveInferenceAgent(config)
            logger.info("ActiveInferenceActionAdapter: Agent initialized")
        except Exception as e:
            logger.warning(f"ActiveInferenceActionAdapter: Could not initialize agent: {e}")
    
    def select_action(self, loop_state: LoopState) -> AgentAction:
        """
        Select action using Active Inference.
        
        Converts loop state to observations, updates beliefs,
        generates candidates, computes EFE, selects best action.
        """
        self.call_count += 1
        
        if not self.agent:
            return self._fallback_action(loop_state)
        
        try:
            # 1. Update beliefs from gaps
            for gap in loop_state.gaps[:5]:
                gap_id = gap.get("gap_id", f"gap_{id(gap)}")
                obs = self._gap_to_observation(gap)
                self.agent.update_belief(gap_id, obs)
            
            # 2. Register knowledge gaps in agent
            for gap in loop_state.gaps:
                gap_id = gap.get("gap_id", f"gap_{id(gap)}")
                self.agent.knowledge_gaps[gap_id] = 1.0  # High uncertainty
            
            # 3. Build context for agent
            context = {
                "cycle": loop_state.cycle,
                "num_hypotheses": len(loop_state.hypotheses),
                "mycelial_edges": loop_state.mycelial_stats.get("active_edges", 0),
                "field_free_energy": loop_state.field_stats.get("free_energy", 0.0),
            }
            
            # 4. Select action via Active Inference EFE computation
            ai_action, info = self.agent.select_action(context=context)
            
            # 5. Convert to unified AgentAction
            agent_action = AgentAction(
                action_type=self._convert_action_type(ai_action.action_type),
                target=ai_action.target,
                parameters=ai_action.parameters,
                expected_free_energy=info.get("selected_EFE", 0.0),
                information_gain=ai_action.expected_information_gain,
                risk=ai_action.expected_risk,
                confidence=info.get("selection_prob", 0.5),
                source="active_inference"
            )
            
            self.last_action = agent_action
            self.action_history.append(agent_action)
            
            # Limit history
            if len(self.action_history) > 100:
                self.action_history = self.action_history[-50:]
            
            return agent_action
            
        except Exception as e:
            logger.warning(f"ActiveInference select_action failed: {e}")
            return self._fallback_action(loop_state)
    
    def update_after_action(self, action: AgentAction, reward: float) -> None:
        """Update agent after action execution"""
        if self.agent:
            self.agent.timestep += 1
            self.agent.decay_beliefs()
    
    def _gap_to_observation(self, gap: Dict) -> np.ndarray:
        """Convert gap to observation vector for belief update"""
        # Try to use topology_engine.encode if available
        if self.topology_engine:
            try:
                desc = gap.get("description", "unknown gap")
                embedding = self.topology_engine.encode([desc])[0]
                # Resize if needed
                if len(embedding) != self.state_dim:
                    embedding = np.resize(embedding, self.state_dim)
                return embedding.astype(np.float32)
            except Exception:
                pass
        
        # Fallback: random observation
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def _convert_action_type(self, ai_action_type) -> ActionType:
        """Convert ActiveInference ActionType enum to unified ActionType"""
        return _ACTION_TYPE_MAP.get(ai_action_type.name, ActionType.REST)
    
    def _fallback_action(self, loop_state: LoopState) -> AgentAction:
        """Fallback action when AI agent is not available"""
        return AgentAction(
            action_type=ActionType.QUERY_SEARCH,
            target="exploration",
            parameters={"reason": "fallback"},
            source="fallback"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return adapter statistics"""
        return {
            "call_count": self.call_count,
            "agent_available": self.agent is not None,
            "last_action": self.last_action.to_dict() if self.last_action else None,
            "history_length": len(self.action_history)
        }
