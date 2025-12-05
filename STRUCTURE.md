# Alexandria - Project Structure

## 📁 Directory Organization

```
alexandria/
├── 📂 core/                    # Core system modules
│   ├── agents/                 # Action & Critic agents
│   ├── memory/                 # Semantic memory & storage
│   ├── reasoning/              # Mycelial, Abduction, Causal reasoning
│   │   └── vqvae/             # VQ-VAE models
│   ├── topology/               # Topology engine & clustering
│   └── utils/                  # Utilities (harvester, LLM, logger)
│
├── 📂 interface/               # Streamlit UI
│   ├── components/             # Reusable UI components
│   ├── pages/                  # Dashboard pages
│   └── app.py                  # Main application
│
├── 📂 scripts/                 # Utility scripts
│   ├── Training/               # train_*.py
│   ├── Testing/                # test_*.py
│   ├── Analysis/               # analyze_*.py
│   └── ingestion/              # cycle_harvest.py, mass_ingest.py
│
├── 📂 tests/                   # Test suite
│
├── 📂 data/                    # Data directory
│   ├── library/                # Raw documents
│   ├── lancedb/                # Vector database
│   ├── mycelial_state.pkl      # Mycelial network state
│   └── monolith_v3_fineweb.pt  # VQ-VAE Model (Modified Wiki)
│
├── 📂 docs/                    # Documentation
│   └── modules/               # Component docs
│
├── 📄 README.md               # Main documentation
├── 📄 requirements.txt        # Python dependencies
├── 📄 config.py               # System configuration
├── 📄 Dockerfile              # Docker containerization
└── 📄 LICENSE                 # MIT License
```

## 🎯 Core Modules

### Memory (`core/memory/`)
- `semantic_memory.py` (488 lines) - Multi-modal indexing
- `storage.py` (135 lines) - LanceDB wrapper
- `v11_vision_encoder.py` (585 lines) - Image processing

### Reasoning (`core/reasoning/`)
- `mycelial_reasoning.py` (800 lines) - Hebbian learning network
- `abduction_engine.py` (999 lines) - Hypothesis generation
- `causal_reasoning.py` (428 lines) - Causal graph construction
- `neural_learner.py` (355 lines) - Self-learning module
- `vqvae/model_wiki.py` (108 lines) - Wiki-trained VQ-VAE ✨

### Agents (`core/agents/`)
- `action/` - **Refactored modular structure (v2.0.0)** ✨
  - `__init__.py` - Public API exports
  - `types.py` - Enums & dataclasses (ActionType, ActionStatus, EvidenceType)
  - `security_controller.py` - API validation, rate limiting, audit logs
  - `parameter_controller.py` - System parameter management
  - `agent.py` - Main orchestrator (ActionAgent class)
  - `test_simulator.py` - Hypothesis testing simulations
  - `evidence_registrar.py` - Evidence registration in SFS
  - `execution/` - Specialized action executors
    - `api_executor.py` - HTTP API calls
    - `model_executor.py` - ML model training
    - `data_executor.py` - Synthetic data generation
    - `simulation_executor.py` - Simulations & config changes
- `action_agent.py` - **Deprecated wrapper** (backward compatibility)
- `bridge_agent.py` (313 lines) - Knowledge gap bridging
- `critic_agent.py` (312 lines) - Hypothesis criticism
- `oracle.py` (267 lines) - Knowledge oracle

### Topology (`core/topology/`)
- `topology_engine.py` (502 lines) - Semantic space management

## 📊 Key Files

### Models
- **monolith_v3_fineweb.pt** (11.3 MB) - Production VQ-VAE
  - Modified Wiki Training (V3.1)
  - Orthogonal heads (No hub dominance)
  - 100% codebook usage

### State
- **mycelial_state.pkl** - Mycelial network weights
  - Sparse graph representation
  - Hebbian connections

### Configuration
- **config.py** - System settings
- **.env** - API keys & secrets (git-ignored)
- **requirements.txt** - 25 dependencies



## 🚀 Entry Points

### UI
```bash
streamlit run interface/app.py
```

### Scripts
```bash
# Index documents
python scripts/mass_ingest.py --directory ./papers

# Train mycelial network
python scripts/train_mycelial.py --limit 10000

# Visualize network
python scripts/visualize_mycelial.py

# Run collision experiments
python scripts/collide.py --source "AI" --target "Physics"
```

### API
```bash
# Start FastAPI server
uvicorn core.api:app --reload
```

## 📝 Development

### Testing
```bash
pytest tests/ -v
```

### Code Quality
```bash
black core/ scripts/ tests/
flake8 core/ scripts/ tests/
```

## 🔄 Data Flow

```
Documents → core/memory → LanceDB → core/reasoning/vqvae → Mycelial Network
                                                              ↓
User Query ← Results ← core/reasoning/mycelial_reasoning ← Propagation
```

---

**Last Updated**: 2025-12-04  
**Version**: 3.1.1  
**Structure**: Cleaned and optimized for production  
**Recent Changes**: Action Agent refactored to modular structure (v2.0.0)
