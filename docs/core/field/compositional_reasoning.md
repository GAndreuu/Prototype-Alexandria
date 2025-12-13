# 🧠 CompositionalReasoner

**Module**: `core/field/compositional_reasoning.py`  
**Lines**: 1091  
**Purpose**: Compositional reasoning via geodesic traversal with residual accumulation.

---

## Overview

Vector is **TRANSFORMED** during traversal, not just transported. Nodes contribute semantically.

---

## Dependencies

| Import | Purpose |
|--------|---------|
| `numpy` | Vector operations |
| `.geodesic_flow` | GeodesicFlow for paths |
| `reasoning.mycelial_reasoning` | Hebbian residuals (optional) |

```
final_vector = start_vector + Σ residuals_along_path
```

Where residuals come from Hebbian connections, attention weights, or field gradients.

---

## ⚙️ Configuration

```python
@dataclass
class CompositionConfig:
    residual_mode: ResidualMode = HEBBIAN      # Source of residuals
    composition_strategy: CompositionStrategy = ADDITIVE
    residual_scale: float = 0.1                # Contribution scaling
    attention_heads: int = 4
    attention_temperature: float = 1.0
    field_coupling: float = 0.5
    momentum_alpha: float = 0.9
    normalize_output: bool = True
    decode_concepts: bool = True
```

### Residual Modes

| Mode | Description |
|------|-------------|
| `HEBBIAN` | Uses MycelialReasoning edge weights |
| `ATTENTION` | Multi-head attention over neighbors |
| `FIELD` | Free energy gradients |
| `PROJECTION` | Tangent space projection |
| `HYBRID` | Combination of modes |

### Composition Strategies

| Strategy | Description |
|----------|-------------|
| `ADDITIVE` | Simple sum of residuals |
| `GATED` | Learned gating of contributions |
| `NORMALIZED` | Layer-normalized accumulation |
| `MOMENTUM` | Momentum-based accumulation |

---

## 🔄 Main Operations

### `reason(start, target)` → CompositionalPath
Core reasoning operation:
1. Compute geodesic path from start to target
2. At each step, compute residual from nearby nodes
3. Accumulate residuals into cumulative vector
4. Return transformed vector + trace

### `reason_chain(start, waypoints)`
Chain reasoning through specified waypoints.

### `analogy(a, b, c)` → (d, path)
Compositional analogy: `a:b :: c:?`
- Implements "king - man + woman = queen" via geodesic

---

## 📦 Output Structures

### CompositionalPath
```python
@dataclass
class CompositionalPath:
    start_vector: np.ndarray
    target_vector: np.ndarray
    cumulative_vector: np.ndarray    # Final transformed result
    points: np.ndarray               # Path coordinates
    residuals: np.ndarray            # Accumulated residuals
    path_length: float
    total_residual_magnitude: float
    composition_trace: List[str]     # Concepts visited
    steps: List[CompositionStep]
    dominant_contributions: List[Tuple[int, float]]
```

---

## 🔗 Integration

Requires:
- `VQVAEManifoldBridge` for embedding/decoding
- `MycelialReasoning` (optional) for Hebbian residuals
- `GeodesicFlow` (optional) for path computation

---

**Last Updated**: 2025-12-13  
**Version**: 1.0  
**Status**: Active
