# 🎯 Action Selection Protocol

**Module**: `core/loop/action_selection.py`  
**Lines of Code**: ~76  
**Purpose**: Unified interface for action selection strategies in the cognitive loop.

---

## 🎯 Overview

Defines the **protocol and types** for action selection, enabling pluggable strategies (heuristic, active inference, etc.).

---

## 📦 Core Types

### ActionType (Enum)
```python
class ActionType(Enum):
    QUERY_SEARCH = auto()
    EXPLORE_CLUSTER = auto()
    DEEPEN_TOPIC = auto()
    BRIDGE_CONCEPTS = auto()
    FILL_GAP = auto()
    CONSOLIDATE = auto()
    FOLLOW_CONNECTION = auto()
    REST = auto()
```

### AgentAction
```python
@dataclass
class AgentAction:
    action_type: ActionType
    target: str
    parameters: Dict[str, Any] = {}
    expected_free_energy: float = 0.0
    information_gain: float = 0.0
    risk: float = 0.0
    confidence: float = 0.5
    source: str = "heuristic"  # "heuristic" | "active_inference" | "nemesis"
```

### LoopState
```python
@dataclass
class LoopState:
    cycle: int
    gaps: List[Dict]
    hypotheses: List[Dict]
    mycelial_stats: Dict[str, Any]
    field_stats: Dict[str, Any]
    last_reward: float = 0.0
    recent_actions: List[AgentAction]
```

---

## 🔌 Protocol

### ActionSelectionAdapter
```python
class ActionSelectionAdapter(Protocol):
    def select_action(self, loop_state: LoopState) -> AgentAction: ...
    def update_after_action(self, action: AgentAction, reward: float) -> None: ...
    def get_stats(self) -> Dict[str, Any]: ...
```

Any strategy implementing this protocol can be plugged into `SelfFeedingLoop`.

---

## 🔗 Implementations

| Adapter | Description |
|---------|-------------|
| `ActiveInferenceActionAdapter` | Uses EFE minimization |
| Heuristic (built-in) | Simple gap-based selection |

---

**Last Updated**: 2025-12-13  
**Version**: 1.0  
**Status**: Active
