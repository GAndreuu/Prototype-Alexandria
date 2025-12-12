# 🧪 Alexandria QA 2.0

**Architecture**: Tiered testing strategy for reliability and speed.

## Structure

```
tests/
├── unit/                    # 🚀 Fast, Mocked (No DB/LLM)
│   ├── core/agents/
│   ├── core/field/
│   └── ...
│
├── integration/             # 🐢 Slower, Real I/O (LanceDB, etc)
│   ├── core/memory/
│   ├── core/loop/
│   └── workflows/
│
├── functional/              # 🧪 Real Data / Scenarios
│   ├── test_manifold_runner.py
│   └── test_mycelial_runner.py
│
└── conftest.py              # Global fixtures & Mocks
```

## Running Tests

### 1. Unit Tests (Fast)
```bash
./venv/bin/python -m pytest tests/unit
```

### 2. Integration Tests (Slower)
```bash
./venv/bin/python -m pytest tests/integration
```

### 3. Full Suite (Sequential Runner)
```bash
./venv/bin/python scripts/testing/sequential_runner.py
```
This script runs unit, integration, and functional tests, logging results to `docs/reports/test_logs/`.

## Guidelines
- **Unit**: Mock EVERYTHING external (DB, API). Use `conftest.py` fixtures.
- **Integration**: Use real DB (temp dir) and real components.
- **Functional**: End-to-end flows with real data.
