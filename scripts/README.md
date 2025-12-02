# 📂 Scripts Directory Organization

All scripts organized by category for easy navigation.

---

## 📁 Directory Structure

```
scripts/
├── training/          # Model training scripts
│   ├── train_mycelial.py      (19.6 KB) - Train mycelial network
│   └── train_vqvae.py         (5.3 KB)  - Train VQ-VAE model
│
├── testing/           # Test and validation scripts
│   ├── integration_test.py        (10.3 KB) - Integration tests
│   ├── stress_test.py             (6.5 KB)  - Performance stress tests
│   ├── test_complete_system.py    (3.4 KB)  - Full system test
│   ├── test_integration.py        (1.3 KB)  - Integration test
│   ├── test_model_loading.py      (3.5 KB)  - Model loading test
│   └── test_wiki_quick.py         (876 B)   - Quick wiki model test
│
├── analysis/          # Analysis and visualization
│   ├── analyze_code_semantics.py    (11.7 KB) - Code semantic analysis
│   ├── analyze_db_stats.py          (2.9 KB)  - Database statistics
│   ├── visualize_knowledge_graph.py (6.2 KB)  - Knowledge graph viz
│   └── visualize_mycelial.py        (22.1 KB) - Mycelial network 3D viz
│
├── ingestion/         # Data ingestion and harvesting
│   ├── auto_ingest.py           (2.3 KB) - Auto-ingest from magic folder
│   ├── mass_ingest.py           (3.0 KB) - Bulk directory ingestion
│   ├── harvest_papers.py        (2.4 KB) - arXiv paper harvesting
│   ├── harvest_custom.py        (2.6 KB) - Custom harvesting
│   └── download_papers_bulk.py  (4.4 KB) - Bulk paper download
│
├── demos/             # Demonstration scripts
│   ├── demo_capabilities.py  (5.1 KB) - System capabilities demo
│   ├── demo_full_system.py   (8.5 KB) - Full system demonstration
│   ├── collide.py            (7.4 KB) - Semantic collision demo
│   └── collide_v2.py         (9.3 KB) - Enhanced collision demo
│
├── utilities/         # Utility and helper scripts
│   ├── init_brain.py           (1.3 KB) - Initialize system
│   ├── reset_db.py             (678 B)  - Reset database
│   ├── export_embeddings.py    (2.1 KB) - Export embeddings
│   ├── convert_wiki_weights.py (2.5 KB) - Convert weights
│   ├── count_papers.py         (1.0 KB) - Count indexed papers
│   └── finetune_llm.py         (4.0 KB) - Fine-tune local LLM
│
└── entrypoint.sh      # Docker entrypoint
```

---

## 🎯 Quick Access

### Training
```bash
# Train mycelial network
python scripts/training/train_mycelial.py --limit 10000

# Train VQ-VAE
python scripts/training/train_vqvae.py --epochs 20
```

### Testing
```bash
# Run integration tests
python scripts/testing/integration_test.py

# Stress test system
python scripts/testing/stress_test.py

# Quick wiki model test
python scripts/testing/test_wiki_quick.py
```

### Analysis
```bash
# Visualize mycelial network
python scripts/analysis/visualize_mycelial.py

# Analyze database stats
python scripts/analysis/analyze_db_stats.py

# Knowledge graph visualization
python scripts/analysis/visualize_knowledge_graph.py
```

### Ingestion
```bash
# Auto-ingest (watch magic folder)
python scripts/ingestion/auto_ingest.py

# Bulk ingest directory
python scripts/ingestion/mass_ingest.py --directory ./papers --workers 4

# Harvest from arXiv
python scripts/ingestion/harvest_papers.py --query "machine learning" --max-results 50
```

### Demos
```bash
# Demo system capabilities
python scripts/demos/demo_capabilities.py

# Full system demo
python scripts/demos/demo_full_system.py

# Semantic collision
python scripts/demos/collide.py --source "AI" --target "Physics"
```

### Utilities
```bash
# Initialize system
python scripts/utilities/init_brain.py

# Reset database
python scripts/utilities/reset_db.py

# Export embeddings
python scripts/utilities/export_embeddings.py

# Count papers
python scripts/utilities/count_papers.py
```

---

## 📊 Statistics

```
Total Scripts: 27
├─ Training: 2
├─ Testing: 6
├─ Analysis: 4
├─ Ingestion: 5
├─ Demos: 4
└─ Utilities: 6

Total Size: ~150 KB
Average: ~5.6 KB per script
```

---

## 🔍 Script Categories Explained

### Training
Scripts for training models and networks. These are typically run once or periodically to update models.

### Testing
Validation and performance testing scripts. Use these to verify system functionality and benchmark performance.

### Analysis
Visualization and statistical analysis tools. Great for understanding system behavior and debugging.

### Ingestion
Data ingestion and harvesting utilities. Use these to populate the knowledge base with documents.

### Demos
Demonstration scripts showing system capabilities. Good for new users and presentations.

### Utilities
Helper scripts for system maintenance and configuration. Used for setup, cleanup, and exports.

---

**Last Updated**: 2025-12-02  
**Organization**: By functionality  
**Status**: All 27 scripts organized
