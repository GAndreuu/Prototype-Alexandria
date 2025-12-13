# 📦 Action Agent Types

**Module**: `core/agents/action/types.py`  
**Lines of Code**: 78  
**Purpose**: Type definitions for the Action Agent system.

---

## 🎯 Overview

Contains all enums and dataclasses used across the Action Agent.

---

## 📊 Enums

### ActionType
```python
class ActionType(Enum):
    API_CALL = "api_call"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    MODEL_RETRAIN = "model_retrain"
    DATA_GENERATION = "data_generation"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SIMULATION_RUN = "simulation_run"
    INTERNAL_LEARNING = "internal_learning"
```

### ActionStatus
```python
class ActionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### EvidenceType
```python
class EvidenceType(Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"
```

---

## 📦 Dataclasses

### ActionResult
```python
@dataclass
class ActionResult:
    action_id: str
    action_type: ActionType
    status: ActionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    evidence_generated: bool = False
    evidence_type: Optional[EvidenceType] = None
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
```

### TestHypothesis
```python
@dataclass
class TestHypothesis:
    hypothesis_id: str
    hypothesis_text: str
    source_cluster: int
    target_cluster: int
    test_action: ActionType
    test_parameters: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[ActionResult] = None
    evidence_registered: bool = False
```

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
