# 🎯 Alexandria System Overview

**Comprehensive visual guide to system architecture and data flow**

---

## 🌐 Complete System Architecture

```mermaid
graph TB
    subgraph Users
        U1[Researcher]
        U2[Data Scientist]
        U3[Developer]
    end
    
    subgraph Interface Layer
        UI[Streamlit UI]
        API[FastAPI]
        Scripts[CLI Scripts]
    end
    
    subgraph Memory Layer
        SFS[Semantic Memory]
        LDB[(LanceDB)]
        V11[V11 Vision Encoder]
    end
    
    subgraph Reasoning Layer
        VQ[VQ-VAE<br/>Monolith Modified Wiki]
        MC[Mycelial Network<br/>128K observations]
        Abd[Abduction Engine]
        Caus[Causal Reasoning]
    end
    
    subgraph Intelligence Layer
        Act[Action Agent]
        Top[Topology Engine]
        V2[V2 Learner]
    end
    
    subgraph Infrastructure
        Data[(Data Storage)]
        Models[Model Weights]
        State[Network State]
    end
    
    U1 & U2 & U3 --> UI & API & Scripts
    UI & API & Scripts --> SFS
    SFS --> LDB
    SFS --> V11
    SFS --> Top
    LDB --> VQ
    VQ --> MC
    MC --> Abd
    Abd --> Caus
    Caus --> Act
    Act --> Top
    Act --> V2
    Top --> SFS
    V2 --> MC
    
    SFS -.-> Data
    VQ & MC -.-> Models
    MC -.-> State
    
    style Users fill:#e3f2fd
    style Interface Layer fill:#fff3e0
    style Memory Layer fill:#e8f5e9
    style Reasoning Layer fill:#f3e5f5
    style Intelligence Layer fill:#fce4ec
    style Infrastructure fill:#f5f5f5
```

---

## 📊 Data Flow: Document Ingestion → Reasoning

```mermaid
sequenceDiagram
    participant U as User
    participant SFS as Semantic Memory
    participant Top as Topology Engine
    participant LDB as LanceDB
    participant VQ as VQ-VAE
    participant MC as Mycelial
    
    U->>SFS: upload("paper.pdf")
    SFS->>SFS: extract_text()
    SFS->>SFS: chunk_text(~1000 chars)
    
    loop Each chunk
        SFS->>Top: encode(chunk)
        Top-->>SFS: 384D embedding
        SFS->>LDB: insert(embedding, metadata)
        LDB->>VQ: compress(embedding)
        VQ-->>LDB: [h1, h2, h3, h4] codes
        LDB->>MC: observe(codes)
        MC->>MC: Hebbian update
    end
    
    SFS-->>U: Indexed 47 chunks
```

---

## 🔍 Query Flow: Search → Enhanced Results

```mermaid
sequenceDiagram
    participant U as User
    participant SFS as Semantic Memory
    participant LDB as LanceDB
    participant MC as Mycelial
    participant VQ as VQ-VAE
    
    U->>SFS: retrieve("quantum AI")
    SFS->>SFS: encode_query()
    SFS->>LDB: similarity_search()
    LDB-->>SFS: top-10 results
    
    alt Enhanced with Reasoning
        SFS->>VQ: encode(query_embedding)
        VQ-->>SFS: [163, 74, 22, 91]
        SFS->>MC: propagate(codes, steps=3)
        MC->>MC: activation spreading
        MC-->>SFS: enhanced_codes
        SFS->>LDB: search(enhanced_codes)
        LDB-->>SFS: expanded results
    end
    
    SFS-->>U: ranked results (semantic + reasoning)
```

---

## 🧠 Self-Learning Cycle

```mermaid
graph TB
    A[Knowledge Base] --> B[Causal Reasoning]
    B --> C[Gap Detection]
    C --> D[Abduction Engine]
    D --> E[Hypothesis Generation]
    E --> F[Action Agent]
    F --> G{Validation}
    G -->|Pass| H[Evidence Registry]
    G -->|Fail| I[Rejected]
    H --> J[Neural Consolidation]
    J --> K[V2 Learner]
    K --> L[Update Weights]
    L --> M[Mycelial Network]
    M --> A
    
    style A fill:#4CAF50,color:#fff
    style D fill:#E91E63,color:#fff
    style F fill:#FF9800,color:#fff
    style H fill:#2196F3,color:#fff
    style M fill:#9C27B0,color:#fff
```

---

## 🔗 Module Dependencies

```mermaid
graph TB
    Top[Topology Engine<br/>Embeddings]
    SFS[Semantic Memory<br/>Storage]
    VQ[VQ-VAE<br/>Compression]
    MC[Mycelial<br/>Reasoning]
    Abd[Abduction<br/>Hypotheses]
    Caus[Causal<br/>Graphs]
    Act[Action<br/>Execution]
    V2[V2<br/>Learning]
    
    Top --> SFS
    SFS --> VQ
    VQ --> MC
    MC --> Abd
    Abd --> Caus
    Caus --> Act
    Act --> V2
    V2 --> MC
    Act --> SFS
    Caus --> Top
    
    style Top fill:#2196F3,color:#fff
    style SFS fill:#4CAF50,color:#fff
    style VQ fill:#9C27B0,color:#fff
    style MC fill:#FF9800,color:#fff
    style Abd fill:#E91E63,color:#fff
    style Caus fill:#673AB7,color:#fff
    style Act fill:#FF5722,color:#fff
    style V2 fill:#00BCD4,color:#fff
```

---

## 📈 System Metrics

### Performance (100K documents corpus)

| Metric | Value | Status |
|--------|-------|--------|
| **Indexing Speed** | 1,000 chunks/sec | ✅ |
| **Query Latency (p99)** | <300ms | ✅ |
| **Memory Usage** | 295 MB | ✅ |
| **VQ-VAE Compression** | 96% reduction | ✅ |
| **Mycelial Density** | 0.86% (sparse) | ✅ |
| **Codebook Usage** | 100% | ✅ |

### Capacity

| Resource | Current | Max Tested | Theoretical |
|----------|---------|------------|-------------|
| **Documents** | Variable | 100K | 1M+ |
| **Embeddings** | Variable | 1M vectors | 10M+ |
| **Mycelial Observations** | 128,692 | 500K | Unlimited |
| **Causal Edges** | ~8K | 50K | 100K+ |

---

## 🧩 Module Summary

| Module | Purpose | LOC | Key Feature |
|--------|---------|-----|-------------|
| **Semantic Memory** | Document storage | 488 | Multi-modal indexing |
| **Topology Engine** | Embeddings | 502 | SentenceTransformer |
| **VQ-VAE** | Compression | ~200 | 96% reduction |
| **Mycelial Reasoning** | Learning | 800 | Hebbian 100% codebook |
| **Abduction Engine** | Hypotheses | 999 | Self-learning |
| **Causal Reasoning** | Relationships | 428 | Causal graphs |
| **Action Agent** | Validation | 498 | Evidence registry |

**Total Core Code**: ~4,000 lines

---

## 🚀 Technology Stack

### Core
- **Python**: 3.10+
- **PyTorch**: Neural models
- **NumPy**: Numerical operations

### ML/AI
- **SentenceTransformers**: Embeddings (all-MiniLM-L6-v2)
- **Scikit-learn**: Clustering, dimensionality reduction
- **LanceDB**: Vector database

### Interface
- **Streamlit**: Web UI
- **FastAPI**: REST API
- **Plotly**: Interactive visualizations

### Utilities
- **arXiv API**: Paper harvesting
- **PyPDF**: PDF processing
- **Loguru**: Logging

---

## 📁 File Organization

```
alexandria/
├── core/                     # Core system (4K LOC)
│   ├── memory/              # Semantic Memory, V11 Vision
│   ├── reasoning/           # Mycelial, Abduction, Causal, VQ-VAE
│   ├── agents/              # Action Agent, Critic
│   ├── topology/            # Topology Engine
│   └── utils/               # Harvester, LLM, Logger
│
├── interface/               # Streamlit UI
│   ├── pages/              # 5 dashboard pages
│   └── app.py              # Main app
│
├── scripts/                # Utilities (28 scripts)
│   ├── Training            # train_*.py
│   ├── Testing             # test_*.py
│   └── Analysis            # analyze_*.py
│
├── data/                   # Models & state
│   ├── monolith_v13_wiki_trained.pth  (7.9 MB)
│   ├── mycelial_state.npz             (network state)
│   └── lancedb/                       (vector DB)
│
├── docs/                   # Documentation
│   ├── modules/           # This documentation!
│   └── README.md          # Index
│
└── tests/                  # Test suite (80%+ coverage)
```

---

## 🎯 Typical Workflows

### 1. Researcher: Literature Review

```
1. Upload papers → Semantic Memory
2. System indexes → LanceDB
3. VQ-VAE compresses → 4 bytes/chunk
4. Mycelial observes → learns patterns
5. Query "quantum AI" → enhanced results
6. Abduction generates → new hypotheses
7. Researcher validates → system learns
```

### 2. Data Scientist: Knowledge Discovery

```
1. Bulk ingest dataset → mass_ingest.py
2. System clusters → Topology Engine
3. Causal graph built → relationships emerge
4. Query concepts → find paths
5. Discover latent variables → hidden causes
6. Validate with Action Agent → evidence
```

### 3. Developer: System Extension

```
1. Add new module → core/
2. Integrate with Semantic Memory → index_file()
3. Use Topology for embeddings → encode()
4. Leverage Mycelial for reasoning → reason()
5. Test with pytest → tests/
6. Deploy → Docker
```

---

## 🔮 Roadmap to ASI Local

Current system is **foundation**. Next phases:

### Phase 1: Enhanced Thinking ✅ (Completed)
- ✅ Semantic Memory
- ✅ Mycelial Reasoning
- ✅ VQ-VAE Compression
- ✅ Self-Learning (Abduction)

### Phase 2: Action Layer (Next)
- [ ] Local LLM integration (Llama3)
- [ ] Tool use framework
- [ ] Planning module (MCTS)
- [ ] Code execution sandbox

### Phase 3: Meta-Learning
- [ ] Performance tracking
- [ ] Automated experimentation
- [ ] Curriculum generation
- [ ] Self-modification

### Phase 4: Full ASI
- [ ] Multi-agent coordination
- [ ] Long-term memory
- [ ] Goal-oriented behavior
- [ ] Emergent capabilities

---

## 📞 Support

- **Documentation**: [docs/README.md](./README.md)
- **Module Guides**: [docs/modules/](./modules/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Last Updated**: 2025-12-04
**Version**: 3.1
**Status**: Production-ready cognitive AI system
