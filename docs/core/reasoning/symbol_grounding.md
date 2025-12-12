# 🔗 Symbol Grounding

**Module**: `core/reasoning/symbol_grounding.py`
**Lines of Code**: ~117
**Purpose**: Bridge between symbolic (text) and subsymbolic (neural/graph) representations.

---

## 🎯 Overview

The **SymbolGrounder** class converts free-form text into discrete **Mycelial Graph nodes**. This is the critical interface between:
- Human-readable concepts → (e.g., "machine learning")
- Neural representations → 384D embedding
- Discrete graph nodes → `[(head_0, code_0), (head_1, code_1), ...]`

### Pipeline
```
Text → TopologyEngine (embed) → 384D Vector → VQ-VAE (quantize) → (head, code) Nodes
```

---

## 📊 Core Class

### `SymbolGrounder`
```python
class SymbolGrounder:
    def __init__(
        self,
        topology_engine: Optional[TopologyEngine] = None,
        vqvae_wrapper: Optional[MycelialVQVAE] = None
    )
    
    def ground(self, text: str) -> List[Tuple[int, int]]:
        """Converts text to list of (head, code) tuples."""
        
    def ground_gap(self, gap_description: str) -> List[Tuple[int, int]]:
        """Wrapper for grounding knowledge gap descriptions."""
```

---

## 🎯 Use Cases

### Basic Grounding
```python
from core.reasoning.symbol_grounding import SymbolGrounder

grounder = SymbolGrounder()
nodes = grounder.ground("machine learning")
# nodes = [(0, 12), (1, 55), (2, 128), (3, 9)]
```

### Grounding a Knowledge Gap
```python
gap_desc = "missing connection between causal inference and reinforcement learning"
nodes = grounder.ground_gap(gap_desc)
```

---

## 🔗 Dependencies
- **TopologyEngine**: Provides text → embedding conversion.
- **MycelialVQVAE**: Provides embedding → discrete codes conversion.

**Used By**: `AbductionEngine`, `BridgeAgent`, `ActionAgent`.

---

**Last Updated**: 2025-12-11
**Status**: Production
