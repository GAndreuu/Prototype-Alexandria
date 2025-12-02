# 📖 Alexandria - User Manual

**Complete guide for using the Alexandria cognitive AI system**

---

## 🎯 What is Alexandria?

Alexandria is a **local-first AI system** that helps you:
- 📚 **Store and search** documents intelligently
- 🧠 **Discover connections** between concepts
- 🔮 **Generate insights** automatically
- ⚡ **Learn continuously** from your data

**Key Benefit**: Everything runs on YOUR computer. No cloud, no data sharing, 100% private.

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Installation

```bash
# Clone repository
git clone https://github.com/yourusername/alexandria.git
cd alexandria

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Launch Interface

```bash
streamlit run interface/app.py
```

Open browser to: `http://localhost:8501`

### Step 3: Index Your First Document

1. Go to **Dashboard** tab
2. Click **"Upload Documents"**
3. Select a PDF or text file
4. Wait for indexing to complete
5. Search for content!

---

## 📚 Main Features

### 1. 🗂️ Document Management

#### Uploading Documents

**Supported Formats**:
- ✅ PDF files (.pdf)
- ✅ Text files (.txt, .md)
- ✅ Images (.jpg, .png, .jpeg)

**How to Upload**:

**Method A: Streamlit UI** (Recommended)
1. Open **Dashboard** tab
2. Drag-and-drop files OR click "Browse files"
3. Click **"Index Documents"**
4. Wait for confirmation

**Method B: Bulk Upload** (For many files)
```bash
# Index entire directory
python scripts/mass_ingest.py --directory ./papers --workers 4

# Auto-harvest from arXiv
python scripts/auto_ingest.py --query "machine learning" --max-results 100
```

#### Viewing Indexed Documents

1. Go to **Dashboard** tab
2. Click **"Show Statistics"**
3. See:
   - Total documents indexed
   - Total chunks
   - Storage usage
   - Mycelial network stats

---

### 2. 🔍 Searching

#### Basic Search

1. Go to **Dashboard** tab
2. Enter query in search box: `"quantum computing"`
3. Click **"Search"**
4. View ranked results

**Search Options**:
- **Modality Filter**: 
  - `All` - Search text and images
  - `Text` - Only text documents
  - `Images` - Only images
- **Limit**: Number of results (default: 10)

#### Advanced Search (with Reasoning)

1. Go to **Mycelial Brain** tab
2. Enter query
3. Enable **"Enhanced Search"**
4. System will:
   - Find direct matches
   - Propagate through semantic network
   - Return expanded results with related concepts

**Example**:
```
Query: "neural networks"

Basic Results:
- Document about neural networks
- Document about deep learning

Enhanced Results:
- Document about neural networks
- Document about deep learning
- Document about backpropagation (related!)
- Document about optimization (connected!)
```

---

### 3. 🍄 Mycelial Brain

**What is it?**  
A self-learning network that discovers connections between concepts automatically.

#### Training the Network

**When to train**:
- After adding 100+ new documents
- When you notice search quality declining
- First time using the system

**How to train**:
```bash
# Option 1: Script
python scripts/train_mycelial.py --limit 10000

# Option 2: UI
# Go to "Mycelial Brain" tab → Click "Train Network"
```

#### Viewing Network Statistics

1. Go to **Mycelial Brain** tab
2. Click **"Show Stats"**
3. See:
   - Total observations
   - Active connections
   - Network density
   - Hub codes (most connected concepts)

#### Visualizing the Network

1. Go to **Mycelial Brain** tab
2. Click **"Visualize Network"**
3. Explore 3D interactive graph:
   - **Nodes**: Semantic codes
   - **Edges**: Learned connections
   - **Size**: Connection strength

---

### 4. 🕸️ Knowledge Graph

**What is it?**  
A causal map showing how concepts influence each other.

#### Building the Graph

1. Go to **Knowledge Graph** tab
2. Click **"Build Graph"**
3. Wait for analysis to complete

#### Exploring Relationships

**Find Causal Path**:
1. Enter **Source concept**: "Machine Learning"
2. Enter **Target concept**: "Image Recognition"
3. Click **"Find Path"**
4. View causal chain:
   ```
   Machine Learning → Deep Learning → CNNs → Image Recognition
   ```

**Discover Hidden Causes**:
1. Enter **Concept A**: "Ice Cream Sales"
2. Enter **Concept B**: "Drowning Deaths"
3. Click **"Find Latent Variables"**
4. Result: "Summer Temperature" (hidden cause!)

---

### 5. 🔮 Abduction Engine

**What is it?**  
Automatically generates hypotheses to fill knowledge gaps.

#### Running Abduction Cycle

1. Go to **Abduction** tab
2. Click **"Detect Knowledge Gaps"**
3. Review detected gaps
4. Click **"Generate Hypotheses"** (max 10)
5. Validate hypotheses:
   - ✅ Green = Validated
   - ❌ Red = Rejected
6. Click **"Consolidate Knowledge"** for validated ones

#### Understanding Results

**Knowledge Gap Types**:
- **Orphaned Cluster**: Concept with no connections
- **Missing Connection**: Should be connected but isn't
- **Broken Chain**: Incomplete causal path

**Hypothesis Validation**:
- **Semantic Coherence**: Does it make sense?
- **Co-occurrence**: Do concepts appear together?
- **Sequential Patterns**: Does A → B in documents?

---

### 6. 💥 Semantic Collider

**What is it?**  
Find surprising connections between different domains.

#### Running a Collision

1. Go to **Collider** tab
2. Enter **Domain A**: "Artificial Intelligence"
3. Enter **Domain B**: "Quantum Physics"
4. Click **"Collide!"**
5. View results:
   - **Bridge codes**: Concepts connecting both
   - **Collision strength**: How related they are
   - **Common documents**: Papers mentioning both

**Example Output**:
```
🌉 Bridge Concepts Found:
1. "Optimization" (strength: 0.87)
2. "Algorithms" (strength: 0.74)
3. "Computation" (strength: 0.68)

📄 Common Documents: 5 papers
```

---

## 🛠️ Advanced Features

### Batch Processing

**Index entire directories**:
```bash
python scripts/mass_ingest.py \
    --directory ./research_papers \
    --doc-type SCI \
    --workers 4
```

**Auto-harvest from arXiv**:
```bash
python scripts/auto_ingest.py \
    --query "transformer models" \
    --max-results 50
```

### Custom Mycelial Configuration

Edit `core/reasoning/mycelial_reasoning.py`:

```python
config = MycelialConfig(
    learning_rate=0.01,      # How fast it learns
    decay_rate=0.001,        # How fast it forgets
    propagation_steps=5,     # Reasoning depth
    connection_threshold=0.05 # Minimum strength to save
)
```

### Exporting Data

**Export knowledge graph**:
```python
from core.reasoning.causal_reasoning import CausalEngine

engine = CausalEngine()
engine.export_graph("my_knowledge_graph.json")
```

**Export mycelial network**:
```python
from core.reasoning.mycelial_reasoning import MycelialReasoning

network = MycelialReasoning()
network.save_state("mycelial_backup.npz")
```

---

## 🎓 Best Practices

### 1. Document Organization

**Do**:
- ✅ Group related papers in subdirectories
- ✅ Use descriptive filenames
- ✅ Keep PDFs text-searchable (not scanned)

**Don't**:
- ❌ Mix personal and research documents
- ❌ Use special characters in filenames
- ❌ Upload duplicate files

### 2. Training Strategy

**Small Corpus** (<1000 docs):
- Train mycelial network: **Every 100 new docs**
- Rebuild knowledge graph: **Weekly**

**Large Corpus** (>10K docs):
- Train mycelial network: **Every 500 new docs**
- Rebuild knowledge graph: **Monthly**

### 3. Search Tips

**For Best Results**:
- Use specific terms: "neural network architectures" > "AI"
- Try multiple phrasings
- Enable enhanced search for exploratory queries
- Use modality filters to narrow down

**Query Examples**:
- ✅ Good: "transformer attention mechanism implementation"
- ❌ Bad: "AI stuff"

### 4. Performance Optimization

**If searches are slow**:
1. Check corpus size (use `get_stats()`)
2. Reduce query limit (10 → 5)
3. Disable enhanced search for simple queries
4. Consider archiving old documents

**If indexing is slow**:
1. Use batch processing scripts
2. Increase worker count: `--workers 8`
3. Disable OCR for text-based PDFs
4. Close other applications (free RAM)

---

## ❓ Troubleshooting

### "Failed to load model"

**Cause**: Missing wiki-trained VQ-VAE model  
**Solution**:
```bash
# Download model (if available)
# Or use fallback:
python scripts/init_brain.py
```

### "LanceDB connection error"

**Cause**: Corrupted database  
**Solution**:
```bash
# Backup data
cp -r data/lancedb data/lancedb_backup

# Reinitialize
rm -rf data/lancedb
python scripts/init_brain.py
```

### "Out of memory"

**Cause**: Corpus too large for RAM  
**Solutions**:
1. **Reduce batch size**:
   ```python
   # In scripts/mass_ingest.py
   BATCH_SIZE = 50  # Reduce from 100
   ```

2. **Enable streaming** (future feature)

3. **Upgrade RAM** (8GB → 16GB recommended)

### "Search returns no results"

**Checklist**:
- ✅ Documents indexed? (check stats)
- ✅ Query spelled correctly?
- ✅ Modality filter correct? (Text vs Image)
- ✅ Try broader query terms

### "Mycelial network not learning"

**Solutions**:
1. Check observations count:
   ```python
   network.get_network_stats()
   # If observations == 0, run training
   ```

2. Re-train:
   ```bash
   python scripts/train_mycelial.py --limit 10000
   ```

3. Reset network (last resort):
   ```python
   network.reset()
   ```

---

## 📊 Understanding System Metrics

### Dashboard Statistics

```
📊 System Status:
├─ Total Documents: 1,247
├─ Total Chunks: 15,832
├─ Storage Used: 295 MB
├─ Mycelial Observations: 128,692
└─ Network Density: 0.86%
```

**What this means**:
- **Documents**: Unique files indexed
- **Chunks**: Text segments (~1000 chars each)
- **Storage**: RAM + disk usage
- **Observations**: Times mycelial network learned
- **Density**: % of possible connections (lower = more efficient)

### Mycelial Network Health

**Good Network**:
- ✅ Observations: >10,000
- ✅ Density: <2%
- ✅ Hub codes: 10-30
- ✅ Avg connections/code: 2-5

**Needs Training**:
- ❌ Observations: <1,000
- ❌ Density: <0.1% (too sparse)
- ❌ Hub codes: <5
- ❌ Avg connections/code: <1

---

## 🔐 Privacy & Security

### Data Storage

**All data stays local**:
- ✅ Documents: `data/library/`
- ✅ Vector database: `data/lancedb/`
- ✅ Network state: `data/mycelial_state.npz`
- ✅ Models: `data/*.pth`

**Nothing sent to cloud** (unless you use external APIs like arXiv).

### API Keys

If using external features (arXiv harvesting, etc.):

1. Copy `.env.example` → `.env`
2. Add your API keys:
   ```env
   ARXIV_API_KEY=your_key_here
   ```
3. Keys never leave your machine

### Backups

**Recommended backup strategy**:
```bash
# Weekly backup
cp -r data/ backups/data_$(date +%Y%m%d)

# Or use script
python scripts/backup_system.py
```

---

## 🚀 Advanced Workflows

### Research Assistant

```
1. Upload all papers in your field
2. Train mycelial network
3. Search for specific topics
4. Use abduction to find research gaps
5. Generate hypotheses for new research
```

### Literature Review

```
1. Auto-harvest papers from arXiv
2. Index all documents
3. Build knowledge graph
4. Find causal paths between concepts
5. Export graph for visualization
```

### Knowledge Discovery

```
1. Upload diverse datasets
2. Run semantic collision
3. Discover unexpected connections
4. Validate with action agent
5. Consolidate new knowledge
```

---

## 📞 Support

**Documentation**:
- [System Overview](./SYSTEM_OVERVIEW.md)
- [Module Documentation](./modules/)
- [API Reference](./API_REFERENCE.md)

**Community**:
- GitHub Issues
- Discussions Forum

**Contact**:
- Email: support@alexandria.ai (if available)

---

## 🎯 Next Steps

**Beginner**:
1. ✅ Index 10-20 documents
2. ✅ Practice searching
3. ✅ Train mycelial network
4. ✅ Explore results

**Intermediate**:
1. ✅ Auto-harvest from arXiv
2. ✅ Build knowledge graph
3. ✅ Run semantic collisions
4. ✅ Use abduction engine

**Advanced**:
1. ✅ Custom mycelial config
2. ✅ Integrate with local LLM
3. ✅ Extend with new modules
4. ✅ Contribute to codebase

---

**Last Updated**: 2025-12-02  
**Version**: 1.0  
**System**: Alexandria Cognitive AI  
**Status**: Production
