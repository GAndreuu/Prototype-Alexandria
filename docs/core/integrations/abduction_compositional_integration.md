# 🔮 Abduction Compositional Integration

**Module**: `core/integrations/abduction_compositional_integration.py`
**Lines of Code**: ~836
**Purpose**: Connect the Abduction Engine to compositional reasoning, representing hypotheses as geodesic paths.

---

## 🎯 Overview

This integration reframes **knowledge gaps** as geometric discontinuities and **hypotheses** as geodesic paths that bridge them. A gap is detected when:
1. Curvature is anomalously low (no attractors nearby).
2. Free energy is high (unmapped region).
3. Clusters are geodesically disconnected.

### Key Insight
A hypothesis is no longer just text—it's a **path through the manifold** that could connect two previously unlinked concepts.

---

## 📊 Core Classes

### `GeometricGap`
```python
@dataclass
class GeometricGap:
    gap_id: str
    gap_type: str               # "curvature", "energy", "disconnection"
    location: np.ndarray        # Center of the gap
    source_region: np.ndarray
    target_region: np.ndarray
    energy_barrier: float       # How hard to cross
    geodesic_distance: float
    priority_score: float       # For ranking gaps
    epistemic_value: float      # Information gain if closed
```

### `GeodesicHypothesis`
```python
@dataclass
class GeodesicHypothesis:
    hypothesis_id: str
    hypothesis_text: str
    gap_id: str
    geodesic_path: np.ndarray   # The proposed bridge
    path_energy: float
    curvature_traversed: float
    confidence_score: float
    validation_score: float
```

---

## 🎯 Key Methods

| Method | Description |
|--------|-------------|
| `detect_gaps_geometric(embeddings)` | Finds gaps via curvature, energy, and disconnection analysis |
| `generate_geodesic_hypotheses(gap)` | Creates candidate paths to close a gap |
| `validate_hypothesis(hypothesis, evidence)` | Scores a hypothesis based on path properties |
| `consolidate_hypothesis(hypothesis)` | Deforms the metric to permanently close the gap |

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Metric and geodesic computation.
- **AbductionEngine**: Base hypothesis generation logic.
- **CompositionalReasoner**: For compositional path construction.

---

**Last Updated**: 2025-12-11
**Status**: Production
