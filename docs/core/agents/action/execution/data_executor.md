# 📊 Data Executor

**Module**: `core/agents/action/execution/data_executor.py`  
**Lines of Code**: 152  
**Purpose**: Handles synthetic data generation using scikit-learn.

---

## 🎯 Overview

Generates synthetic data for testing and simulation purposes.

---

## 📦 Function: `execute_data_generation`

```python
def execute_data_generation(
    parameters: Dict[str, Any],
    sfs_path: Path,
    action_id: str
) -> ActionResult
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_type` | str | "random" | Type of data to generate |
| `size` | int | 1000 | Number of samples |
| `dimensions` | int | 384 | Feature dimensions |
| `seed` | int | 42 | Random seed |

### Data Types
| Type | Description |
|------|-------------|
| `random` | Clustered data with 5 centers |
| `synthetic_v11` | Hierarchical clusters simulating V11 output |
| `text_embeddings` | Semantic-like embeddings with t-SNE |
| `causal_clusters` | Data with causal structure |

---

## 📊 Output
```python
{
    "data_type": "synthetic_v11",
    "data_file": "data/synthetic_data_1234.npy",
    "size": 1000,
    "dimensions": 384,
    "data_shape": [1000, 384],
    "file_size_mb": 1.46
}
```

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
