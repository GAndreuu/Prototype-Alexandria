# 🎮 Simulation Executor

**Module**: `core/agents/action/execution/simulation_executor.py`  
**Lines of Code**: 189  
**Purpose**: Handles simulations, internal learning, and system configuration changes.

---

## 🎯 Overview

Contains multiple executor functions for simulation-related actions.

---

## 📦 Functions

### `execute_parameter_adjustment(parameters, param_controller, action_id)`
Adjusts system parameters via ParameterController.

### `execute_config_change(parameters, action_id)`
Changes system configuration via environment variables.

### `execute_simulation(parameters, action_id)`
Runs real simulations with deterministic calculations based on parameter hash:
- Computes convergence, stability, efficiency
- Uses numpy with seeded randomness for reproducibility

### `execute_internal_learning(parameters, v2_learner, action_id)`
Executes internal learning step using V2Learner.

---

## 📊 Simulation Output
```python
{
    "simulation_name": "custom",
    "metrics": {
        "convergence_rate": 0.85,
        "stability": 0.92,
        "efficiency": 0.78
    },
    "real_simulation": True
}
```

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
