# 🧪 Tests Directory

**Purpose**: Automated test suite for Alexandria.

## Structure
```
tests/
├── conftest.py              # Pytest fixtures
├── data/                    # Test data files
├── test_action_agent_refactor.py
├── test_active_inference_integration.py
├── test_core.py
├── test_executor_real.py
├── test_field.py            # [NEW]
├── test_field_real.py       # [NEW]
├── test_field_reduction.py  # [NEW]
├── test_field_simple.py     # [NEW]
├── test_geodesic_bridge.py  # [NEW]
├── test_model_loading.py    # [NEW]
├── test_mycelial.py
├── test_mycelial_reasoning.py # [NEW]
├── test_predictive_coding.py  # [NEW]
├── test_storage.py
├── test_symbol_grounding.py
├── test_system_integration.py
├── test_v2_cycle.py
└── test_viz.py
```

## Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_mycelial.py -v

# With coverage
python -m pytest tests/ --cov=core --cov-report=html
```

## Test Categories
| Pattern | Description |
|---------|-------------|
| `test_*_integration.py` | Integration tests |
| `test_field*.py` | Field/manifold tests |
| `test_*_real.py` | Tests with real data |

---

**Last Updated**: 2025-12-11
