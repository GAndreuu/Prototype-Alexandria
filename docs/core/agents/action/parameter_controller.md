# ⚙️ ParameterController

**Module**: `core/agents/action/parameter_controller.py`  
**Lines of Code**: 92  
**Purpose**: Parameter management for the Action Agent system.

---

## 🎯 Overview

Handles system parameter adjustments, validation, and history tracking.

---

## 🏗️ Class: ParameterController

```python
class ParameterController:
    def __init__(self)
```

### Supported Parameters

| Parameter | Min | Max | Default |
|-----------|-----|-----|---------|
| `V11_BETA` | 0.1 | 10.0 | 1.0 |
| `V11_LEARNING_RATE` | 0.0001 | 0.1 | 0.001 |
| `V11_BATCH_SIZE` | 16 | 128 | 32 |
| `V11_EPOCHS` | 10 | 200 | 50 |
| `SFS_CHUNK_SIZE` | 256 | 2048 | 512 |
| `SFS_THRESHOLD` | 0.1 | 1.0 | 0.5 |
| `CAUSAL_VARIANCE_THRESHOLD` | 0.01 | 1.0 | 0.1 |
| `CAUSAL_MIN_EDGE_WEIGHT` | 0.01 | 0.5 | 0.05 |

---

## 🔄 Methods

### `adjust_parameter(param_name, new_value) → bool`
Adjusts system parameter:
1. Validates parameter exists
2. Checks min/max range
3. Records in history
4. Updates environment variable
5. Returns success status

### `reset_parameter(param_name) → bool`
Resets parameter to default value.

### `get_parameter(param_name) → Any`
Returns current value of parameter.

### `get_parameter_history(param_name) → List[Dict]`
Returns history of changes for parameter.

---

## 📊 History Entry Format
```python
{
    "timestamp": "2025-12-13T...",
    "parameter": "V11_BETA",
    "old_value": 1.0,
    "new_value": 2.5,
    "change_type": "adjustment"
}
```

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
