"""
Action Selection Protocol and Types
===================================

Unified interface for action selection strategies (heuristic, active inference, etc.)
"""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto


class ActionType(Enum):
    """Unified action types for the cognitive loop"""
    QUERY_SEARCH = auto()
    EXPLORE_CLUSTER = auto()
    DEEPEN_TOPIC = auto()
    BRIDGE_CONCEPTS = auto()
    FILL_GAP = auto()
    CONSOLIDATE = auto()
    FOLLOW_CONNECTION = auto()
    REST = auto()


@dataclass
class AgentAction:
    """Unified action representation from any selector"""
    action_type: ActionType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_free_energy: float = 0.0
    information_gain: float = 0.0
    risk: float = 0.0
    confidence: float = 0.5
    source: str = "heuristic"  # "heuristic" | "active_inference" | "nemesis"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "target": self.target,
            "parameters": self.parameters,
            "expected_free_energy": self.expected_free_energy,
            "information_gain": self.information_gain,
            "risk": self.risk,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class LoopState:
    """State passed to action selector"""
    cycle: int
    gaps: List[Dict] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    mycelial_stats: Dict[str, Any] = field(default_factory=dict)
    field_stats: Dict[str, Any] = field(default_factory=dict)
    last_reward: float = 0.0
    recent_actions: List[AgentAction] = field(default_factory=list)


class ActionSelectionAdapter(Protocol):
    """Protocol for action selection strategies"""
    
    def select_action(self, loop_state: LoopState) -> AgentAction:
        """Select next action based on loop state"""
        ...
    
    def update_after_action(self, action: AgentAction, reward: float) -> None:
        """Update internal state after action execution"""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Return adapter statistics"""
        ...
