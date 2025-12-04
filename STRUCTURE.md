# Alexandria - Project Structure

## 📁 Directory Organization

```
alexandria/
├── 📂 core/                    # Core system modules
│   ├── agents/                 # Action & Critic agents
│   ├── memory/                 # Semantic memory & storage
│   ├── reasoning/              # Mycelial, Abduction, Causal reasoning
│   │   └── vqvae/             # VQ-VAE models (MonolithV13, MonolithWiki)
│   ├── topology/               # Topology engine & clustering
│   └── utils/                  # Utilities (harvester, LLM, logger)
│
├── 📂 interface/               # Streamlit UI
│   ├── components/             # Reusable UI components
│   ├── pages/                  # Dashboard pages
│   └── app.py                  # Main application
│
├── 📂 scripts/                 # Utility scripts
│   ├── Training:
│   │   ├── train_mycelial.py
│   │   └── train_vqvae.py
│   ├── Testing:
│   │   ├── integration_test.py
│   │   ├── stress_test.py
│   │   └── test_model_loading.py
│   ├── Analysis:
│   │   ├── analyze_*.py
│   │   └── visualize_*.py
│   └── Automation:
│       ├── auto_ingest.py
│       └── mass_ingest.py
│
├── 📂 tests/                   # Test suite
│   ├── test_core.py
│   ├── test_mycelial.py
│   ├── test_storage.py
│   ├── test_v2_cycle.py
│   └── test_viz.py
│
├── 📂 data/                    # Data directory
│   ├── library/                # Raw documents
│   ├── lancedb/                # Vector database
│   ├── mycelial_state.npz      # Mycelial network state
│   ├── monolith_v13_trained.pth          # Old VQ-VAE (384D)
│   ├── monolith_v13_wiki_trained.pth     # Wiki VQ-VAE (512D) ✨
│   └── monolith_v13_wiki_codebooks.npz   # Wiki codebooks
│
├── 📂 reports/                 # Generated reports & visualizations
│   ├── collision_report.txt
│   ├── network_viz_3d.html
│   └── system_health_dashboard.png
│
├── 📂 docs/                    # Documentation
│
├── 📂 archive/                 # Deprecated/old files
│   ├── next_passo_old/        # Old training experiments
│   └── README_old.md          # Previous README
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
- `action_agent.py` (498 lines) - Action execution & validation
- `critic_agent.py` (312 lines) - Hypothesis criticism
- `oracle.py` (267 lines) - Knowledge oracle

### Topology (`core/topology/`)
- `topology_engine.py` (502 lines) - Semantic space management

## 📊 Key Files

### Models
- **monolith_v13_wiki_trained.pth** (7.9 MB) - Production VQ-VAE
  - 512D latent space
  - 100% codebook usage
  - Trained on WikiText
  - Power-law distribution (α=1.6)

### State
- **mycelial_state.npz** - Mycelial network weights
  - 128K+ observations
  - 2,252 active connections
  - <1% density

### Configuration
- **config.py** - System settings
- **.env** - API keys & secrets (git-ignored)
- **requirements.txt** - 25 dependencies

## 🗑️ Archived (Not in Use)

Files moved to `archive/` folder:
- `next_passo_old/` - Old training experiments
- `monolith_wikitext_real_extracted/` - Raw training data
- `README_old.md` - Previous documentation

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

**Last Updated**: 2025-12-01  
**Version**: 1.0  
**Structure**: Production-ready, organized, archived legacy code
