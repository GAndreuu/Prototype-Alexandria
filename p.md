# 📚 Alexandria Core Documentation Breakdown



# 📄 FILE: docs/core/README.md
================================================================================

# 📦 Core - Documentação dos Módulos

**Documentação técnica do núcleo do sistema Alexandria**

---

## 🗂️ Estrutura

Esta pasta espelha a estrutura de `core/` no código-fonte:

```
docs/core/
├── agents/           → Agentes do sistema
│   ├── action_agent.md
│   ├── bridge_agent.md
│   ├── critic_agent.md
│   └── neural_oracle.md
│
├── field/            → [NOVO] Campo Pré-Estrutural
│   └── README.md     → Geometria diferencial para cognição
│
├── learning/         → Módulos de aprendizado cognitivo
│   ├── README.md
│   ├── active_inference.md
│   ├── free_energy.md
│   ├── predictive_coding.md
│   ├── meta_hebbian.md
│   ├── integration_layer.md
│   ├── profiles.md
│   └── NEMESIS_MANUAL.md
│
├── loop/             → Self-Feeding Loop
│   ├── README.md
│   ├── self_feeding_loop.md
│   ├── nemesis_integration.md
│   └── feedback_collector.md
│
├── memory/           → Sistemas de memória
│   ├── semantic_memory.md
│   ├── storage.md
│   └── v11_vision_encoder.md
│
├── reasoning/        → Motores de raciocínio
│   ├── abduction_engine.md
│   ├── causal_reasoning.md
│   ├── mycelial_reasoning.md
│   ├── neural_learner.md
│   └── vqvae.md
│
├── topology/         → Motor de embeddings
│   └── topology_engine.md
│
├── integrations/     → [NOVO] Integrações Unificadas
│   └── alexandria_unified.md
│
└── utils/            → Utilitários
    └── README.md
```

---

## 📊 Módulos por Categoria

### 🤖 Agents (`agents/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Action Agent | [action_agent.md](agents/action_agent.md) | Execução de ações e registro de evidências |
| Bridge Agent | [bridge_agent.md](agents/bridge_agent.md) | Identificação e preenchimento de gaps |
| Critic Agent | [critic_agent.md](agents/critic_agent.md) | Validação científica via Gemini |
| Neural Oracle | [neural_oracle.md](agents/neural_oracle.md) | Síntese híbrida (local + cloud) |

### 🌌 Field (`field/`) - NOVO

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| PreStructuralField | [README.md](field/README.md) | Campo geométrico contínuo |
| DynamicManifold | [README.md](field/README.md#1-dynamicmanifold) | Variedade com dimensão variável |
| RiemannianMetric | [README.md](field/README.md#2-riemannianmetric) | Métrica que deforma |
| FreeEnergyField | [README.md](field/README.md#3-freeenergyfield) | Campo F(x) = E - T·S |
| GeodesicFlow | [README.md](field/README.md#4-geodesicflow) | Propagação geodésica |
| CycleDynamics | [README.md](field/README.md#5-cycledynamics) | Expansão→Config→Compressão |

### 🧠 Learning (`learning/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Active Inference | [active_inference.md](learning/active_inference.md) | Agente autônomo de exploração |
| Free Energy | [free_energy.md](learning/free_energy.md) | Princípio unificador |
| Predictive Coding | [predictive_coding.md](learning/predictive_coding.md) | Inferência por minimização de erro |
| Meta-Hebbian | [meta_hebbian.md](learning/meta_hebbian.md) | Plasticidade adaptativa |
| Integration Layer | [integration_layer.md](learning/integration_layer.md) | Conexão entre módulos |
| Profiles | [profiles.md](learning/profiles.md) | Personalidades cognitivas |
| NEMESIS Manual | [NEMESIS_MANUAL.md](learning/NEMESIS_MANUAL.md) | Active Inference + Predictive Coding |

### 🔄 Loop (`loop/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Self-Feeding Loop | [self_feeding_loop.md](loop/self_feeding_loop.md) | Orquestrador do ciclo cognitivo |
| Nemesis Integration | [nemesis_integration.md](loop/nemesis_integration.md) | Cérebro executivo (Active Inference) |
| Feedback Collector | [feedback_collector.md](loop/feedback_collector.md) | Coleta de reward e treinamento |

### 💾 Memory (`memory/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Semantic Memory | [semantic_memory.md](memory/semantic_memory.md) | Multi-modal indexing (texto + imagens) |
| Storage | [storage.md](memory/storage.md) | LanceDB vetorial persistente |
| Vision Encoder | [v11_vision_encoder.md](memory/v11_vision_encoder.md) | VQ-VAE hierárquico visual |

### 🔬 Reasoning (`reasoning/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Abduction Engine | [abduction_engine.md](reasoning/abduction_engine.md) | Geração automática de hipóteses |
| Causal Reasoning | [causal_reasoning.md](reasoning/causal_reasoning.md) | Grafos causais e inferência temporal |
| Mycelial Reasoning | [mycelial_reasoning.md](reasoning/mycelial_reasoning.md) | Rede Hebbian com 638k conexões |
| Neural Learner | [neural_learner.md](reasoning/neural_learner.md) | V2 adapter para VQ-VAE |
| VQ-VAE | [vqvae.md](reasoning/vqvae.md) | Compressão neural 96% |

### 🌐 Topology (`topology/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Topology Engine | [topology_engine.md](topology/topology_engine.md) | Embeddings 384D (SentenceTransformer) |

### 🛠️ Utils (`utils/`)

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| Harvester | [README.md](utils/README.md#harvester) | Colheita automática de papers |
| Local LLM | [README.md](utils/README.md#local-llm) | TinyLlama local para RAG |
| Logger | [README.md](utils/README.md#logger) | Logging estruturado |

---

## 🔗 Mapa de Dependências

```mermaid
graph TB
    T[topology/topology_engine] --> M[memory/semantic_memory]
    M --> V[reasoning/vqvae]
    V --> MC[reasoning/mycelial_reasoning]
    MC --> A[reasoning/abduction_engine]
    A --> C[reasoning/causal_reasoning]
    C --> AA[agents/action_agent]
    AA --> M
    
    A --> B[agents/bridge_agent]
    A --> CR[agents/critic_agent]
    CR --> A
    
    M --> O[agents/neural_oracle]
    
    FE[learning/free_energy] --> AI[learning/active_inference]
    FE --> PC[learning/predictive_coding]
    AI --> MH[learning/meta_hebbian]
    PC --> MH
    MH --> IL[learning/integration_layer]
    
    style T fill:#2196F3,color:#fff
    style M fill:#4CAF50,color:#fff
    style V fill:#9C27B0,color:#fff
    style MC fill:#FF9800,color:#fff
    style A fill:#E91E63,color:#fff
    style FE fill:#9C27B0,color:#fff
```

---

## 📈 Estatísticas

| Categoria | Módulos | Documentados | Cobertura |
|-----------|---------|--------------|-----------|
| Agents | 4 | 4 | ✅ 100% |
| Field | 6 | 6 | ✅ 100% |
| Integrations | 8 | 1 | ⚠️ 12.5% |
| Learning | 7 | 7 | ✅ 100% |
| Loop | 6 | 6 | ✅ 100% |
| Memory | 3 | 3 | ✅ 100% |
| Reasoning | 5 | 5 | ✅ 100% |
| Topology | 1 | 1 | ✅ 100% |
| Utils | 3 | 3 | ✅ 100% |
| **Total** | **43** | **36** | **~84%** |

---

**Última atualização**: 2025-12-11


---


# 📄 FILE: docs/core/agents/action_agent.md
================================================================================

# ⚡ Action Agent

**Module**: `core/agents/action/`  
**Lines of Code**: ~500+ (Modular)  
**Purpose**: Execute actions to validate hypotheses, register evidence, and control system parameters.

---

## 🎯 Overview

The Action Agent is Alexandria's **"hands"** - it executes actions in the real world (or simulation) to test hypotheses generated by the Abduction Engine. It bridges the gap between **theory** (hypotheses) and **practice** (validation).

It has been refactored into a modular system with dedicated controllers for security, parameters, and simulation.

### Core Principle

**Test, Don't Guess**

```
Hypothesis: "Increasing V11_BETA improves clustering density"
Action: PARAMETER_ADJUSTMENT (V11_BETA = 2.5)
Result: Density increased by 15%
Verdict: ✓ Validated (Supporting Evidence)
```

---

## 🏗️ Architecture

```mermaid
graph TD
    AA[Action Agent] --> SC[Security Controller]
    AA --> PC[Parameter Controller]
    AA --> TS[Test Simulator]
    AA --> ER[Evidence Registrar]
    
    SC -->|Validates| API[API Calls]
    PC -->|Manages| SysParams[System Parameters]
    TS -->|Runs| Sims[Simulations]
    ER -->|Indexes| SFS[Semantic Memory]
```

### Components

1.  **ActionAgent** (`agent.py`): The main orchestrator.
2.  **SecurityController** (`security_controller.py`): Enforces rate limits and validates API calls.
3.  **ParameterController** (`parameter_controller.py`): Manages safe adjustment of system parameters (e.g., `V11_BETA`).
4.  **TestSimulator** (`test_simulator.py`): Runs specialized simulations (e.g., VQ-VAE accuracy tests).
5.  **EvidenceRegistrar** (`evidence_registrar.py`): Formats and indexes action results into the Semantic File System.

---

## 🎬 Action Types

The agent supports the following `ActionType` enum values:

### 1. PARAMETER_ADJUSTMENT
**Purpose**: Modify system parameters to test causal effects.
- **Target**: `ParameterController`
- **Example**: Changing learning rates, thresholds, or VQ-VAE beta.

### 2. SIMULATION_RUN
**Purpose**: Run synthetic experiments to validate hypotheses without affecting production data.
- **Target**: `TestSimulator`
- **Example**: `v11_accuracy_test` to verify parameter impact.

### 3. API_CALL
**Purpose**: query external systems (controlled by Security Controller).
- **Target**: External APIs (arXiv, etc.) - *Note: Literature search logic is often handled by Bridge Agent, but executed here.*

### 4. MODEL_RETRAIN
**Purpose**: Trigger retraining of internal models (e.g., V2 Learner).

### 5. DATA_GENERATION
**Purpose**: Generate synthetic data for stress testing or simulation.

### 6. INTERNAL_LEARNING
**Purpose**: Consolidate validated knowledge into the neural weights.

---

## 🔄 Execution Pipeline

```mermaid
sequenceDiagram
    participant Abd as Abduction Engine
    participant AA as Action Agent
    participant SC as Security Controller
    participant Exec as Executor
    participant ER as Evidence Registrar
    
    Abd->>AA: test_hypothesis(hypothesis)
    AA->>SC: check_rate_limit()
    
    alt Allowed
        SC-->>AA: OK
        AA->>Exec: execute_action(params)
        Exec-->>AA: ActionResult
        
        AA->>AA: evaluate_result()
        AA->>ER: register_evidence(result)
        ER-->>AA: evidence_id
        
        AA-->>Abd: TestHypothesis(result)
    else Blocked
        SC-->>AA: Rate Limit Exceeded
        AA-->>Abd: Failed
    end
```

---

## 📊 Core Classes

### ActionResult

```python
@dataclass
class ActionResult:
    action_id: str
    action_type: ActionType
    status: ActionStatus          # PENDING, EXECUTING, COMPLETED, FAILED
    result_data: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]
    evidence_generated: bool
    evidence_type: EvidenceType   # SUPPORTING, CONTRADICTING, NEUTRAL
```

### TestHypothesis

```python
@dataclass
class TestHypothesis:
    hypothesis_id: str
    test_action: ActionType
    test_parameters: Dict
    expected_outcome: Dict
    result: Optional[ActionResult]
    evidence_registered: bool
```

---

## 🔒 Security & Safety

The **SecurityController** ensures the agent doesn't perform dangerous actions:
- **Rate Limiting**: Prevents API spam (e.g., 10 calls / 5 mins).
- **Domain Whitelisting**: Only allows approved external domains.
- **Audit Logging**: Records all actions for review.

The **ParameterController** ensures stability:
- **Range Checking**: Parameters have defined min/max values.
- **History**: Tracks all changes for rollback capability.

---

## 🎯 Use Cases

### 1. Auto-Tuning VQ-VAE
Hypothesis: "Lowering V11_BETA to 0.5 improves reconstruction accuracy."
1.  **Action**: `SIMULATION_RUN` with `parameter_value=0.5`.
2.  **Simulator**: Runs VQ-VAE test, measures accuracy.
3.  **Result**: Accuracy increases.
4.  **Evidence**: Registered as `SUPPORTING`.
5.  **Follow-up**: `PARAMETER_ADJUSTMENT` to permanently set `V11_BETA = 0.5`.

### 2. Validating Causal Links
Hypothesis: "Concept A causes Concept B."
1.  **Action**: `LITERATURE_SEARCH` (via API_CALL).
2.  **Result**: Found papers confirming link.
3.  **Evidence**: Registered in SFS.
4.  **Consolidation**: `INTERNAL_LEARNING` updates Mycelial weights.

---

**Last Updated**: 2025-12-04
**Version**: 2.0 (Modular Refactor)
**Status**: Active


---


# 📄 FILE: docs/core/agents/bridge_agent.md
================================================================================

# Bridge Agent Documentation

## Overview

The **Bridge Agent** is the metacognitive component of Alexandria, responsible for identifying what the system *doesn't* know and planning actions to acquire that knowledge. It transforms passive "gaps" in the knowledge graph into active research missions.

## Core Concepts

### 1. Knowledge Gap (`KnowledgeGap`)
A formal representation of a missing link in the system's knowledge.
- **Source/Target**: The two concepts that should be connected but aren't.
- **Vector Projection**: A mathematical estimation of where the missing information lies in the semantic space.
- **Relation Type**: The nature of the missing link (e.g., `missing_mechanism`, `missing_application`).

### 2. Bridge Request (`BridgeRequest`)
A concrete plan to acquire the missing knowledge.
- **Semantic Query**: A vector for searching similar concepts.
- **Text Query**: A boolean query for external databases (arXiv, etc.).
- **Bridge Spec**: Symbolic constraints (domain, formalism, etc.).

### 3. Bridge Candidate (`BridgeCandidate`)
A potential document or piece of information that could fill the gap.
- **Impact Score**: How well it connects the source and target.
- **Novelty Score**: How much new information it brings compared to existing memory.

## Workflow

1. **Gap Detection**: The Abduction Engine identifies a `KnowledgeGap`.
2. **Planning**: The Bridge Agent analyzes the gap and generates a `BridgeRequest`.
   - It infers the domain (e.g., "Causal Inference").
   - It determines the missing piece (e.g., "Mathematical Formalism").
   - It constructs a hybrid search query.
3. **Acquisition**: The Action Agent (or user) executes the search.
4. **Evaluation**: The Bridge Agent scores the results (`BridgeCandidate`) based on:
   - **Bridge Similarity**: Does it lie between the source and target vectors?
   - **Novelty**: Is it redundant with current memory?
5. **Integration**: The best candidate is ingested into the system.

## Usage

```python
from core.agents.bridge_agent import BridgeAgent, KnowledgeGap

agent = BridgeAgent()

# 1. Receive a gap from Abduction Engine
gap = KnowledgeGap(...)

# 2. Plan the bridge
request = agent.plan_bridge(gap)
print(f"Searching for: {request.text_query}")

# 3. Evaluate a candidate paper
score = agent.evaluate_candidate(gap, request, candidate_embedding, metadata, memory_vectors)
```

## Integration

The Bridge Agent sits between the **Abduction Engine** (which finds problems) and the **Action Agent** (which executes solutions). It provides the *intelligence* needed to turn a vague sense of "not knowing" into a precise research strategy.


---


# 📄 FILE: docs/core/agents/critic_agent.md
================================================================================

# 🧐 Critic Agent

**Module**: `core/agents/critic_agent.py`  
**Lines of Code**: ~770  
**Purpose**: Evaluate hypotheses, calculate risk scores, and provide self-regulation feedback using Gemini.

---

## 🎯 Overview

The **Critic Agent** (Senior Critic V10) serves as the system's "conscience" and quality control layer. While the Abduction Engine generates hypotheses (creativity), the Critic Agent evaluates them for scientific rigor, plausibility, and safety (scrutiny).

It utilizes **Google Gemini** (via API) to perform deep semantic analysis of hypotheses against retrieved evidence.

### Core Responsibilities

1.  **Hypothesis Evaluation**: Assigns a `TruthScore` (0.0 - 1.0) to generated hypotheses.
2.  **Risk Assessment**: Categorizes hypotheses into Risk Levels (Low, Medium, High, Critical).
3.  **Bias Detection**: Identifies confirmation bias, recency bias, and availability bias in the system's reasoning.
4.  **Self-Regulation**: Suggests adjustments to system parameters (e.g., temperature, variance) based on performance.

---

## 🏗️ Architecture

```mermaid
graph TD
    Hypothesis[Hypothesis] --> Critic[Critic Agent]
    Evidence[Supporting/Contradicting Evidence] --> Critic
    
    Critic -->|Prompts| Gemini[Gemini API]
    Gemini -->|Analysis| Critic
    
    Critic --> Assessment[Critical Assessment]
    
    Assessment -->|Feedback| System[System Parameters]
    Assessment -->|Validation| Abduction[Abduction Engine]
```

---

## 📊 Scoring System

### Truth Score
A float from 0.0 to 1.0 representing the likelihood of the hypothesis being true based on current scientific knowledge.

| Score Range | Category | Description |
|-------------|----------|-------------|
| 0.8 - 1.0 | **FACTUAL** | Well-established scientific fact. |
| 0.6 - 0.79 | **PLAUSIBLE** | Likely true, supported by strong evidence. |
| 0.4 - 0.59 | **SPECULATIVE** | Interesting but unproven; requires more testing. |
| 0.0 - 0.39 | **FALSE** | Contradicted by evidence or scientifically invalid. |

### Risk Level
Determines the safety of acting upon a hypothesis.

- **LOW**: Safe to consolidate immediately.
- **MEDIUM**: Requires standard validation (simulation/search).
- **HIGH**: Requires manual review or extensive multi-modal validation.
- **CRITICAL**: Rejected automatically; potential hallucination or dangerous inference.

---

## 🧠 Self-Regulation Loop

The Critic Agent monitors the system's "health" and suggests parameter adjustments:

1.  **Temperature Control**:
    - If Approval Rate > 80% (Too permissive) → **Decrease Temperature** (Reduce creativity).
    - If Approval Rate < 40% (Too strict) → **Increase Temperature** (Encourage exploration).

2.  **Variance Control**:
    - If Bias Detected > 0.6 → **Increase Variance** (Force diversity in sampling).

---

## 📝 Critical Assessment Structure

Every evaluation produces a `CriticalAssessment` object:

```python
@dataclass
class CriticalAssessment:
    hypothesis_id: str
    truth_score: float
    truth_category: TruthScore
    risk_level: RiskLevel
    evidence_quality: float
    reasoning_coherence: float
    supporting_facts: List[str]
    contradicting_facts: List[str]
    recommendation: str  # "approve", "review", "reject"
```

---

## 🔍 Bias Detection

The agent actively looks for cognitive biases in the system's operation:

- **Confirmation Bias**: Accepting hypotheses just because they match existing clusters.
- **Recency Bias**: Over-weighting recently ingested documents.
- **Availability Bias**: Relying on easily accessible evidence (e.g., only text, ignoring images).

---

## 🔗 Integration

- **Input**: Receives `Hypothesis` from `AbductionEngine`.
- **Context**: Queries `SemanticFileSystem` for supporting/contradicting evidence.
- **Output**: Returns `CriticalAssessment` to `ActionAgent` or `AbductionEngine`.

---

**Last Updated**: 2025-12-04
**Version**: 1.0 (V10 Implementation)
**Status**: Active


---


# 📄 FILE: docs/core/agents/neural_oracle.md
================================================================================

# 🔮 Neural Oracle

**Module**: `core/agents/oracle.py`  
**Lines of Code**: ~200  
**Purpose**: Hybrid natural language processing and synthesis using a "Cortex of Experts" architecture.

---

## 🎯 Overview

The **Neural Oracle** is the system's voice and synthesizer. It implements a **Hybrid Architecture** that combines the speed and privacy of a local LLM with the reasoning power of a cloud-based foundation model.

### The "Cortex of Experts"

1.  **Tactical Expert (Local)**:
    - **Model**: `TinyLlama-1.1B-Chat` (Optimized for CPU/AVX512).
    - **Role**: Fast, factual RAG (Retrieval-Augmented Generation), privacy-sensitive processing.
    - **Latency**: < 100ms.
    - **Cost**: Free.

2.  **Strategic Expert (Cloud)**:
    - **Model**: `Google Gemini 1.5 Flash`.
    - **Role**: Complex reasoning, stylistic refinement, "Semantic Collision" synthesis.
    - **Latency**: ~1-2s.
    - **Cost**: API usage.

---

## 🏗️ Architecture

```mermaid
graph LR
    User[User Query] --> Oracle[Neural Oracle]
    
    Oracle -->|Fast Path| Local[Local LLM<br/>TinyLlama]
    Oracle -->|Slow Path| Cloud[Cloud LLM<br/>Gemini]
    
    Local --> Draft[Factual Draft]
    Draft --> Cloud
    
    Cloud -->|Refines| Final[Final Synthesis]
    
    subgraph "Tactical Layer"
    Local
    end
    
    subgraph "Strategic Layer"
    Cloud
    end
```

---

## 🔄 Synthesis Modes

The Oracle operates in three modes:

### 1. Local Mode (Offline/Fast)
- Uses **only** TinyLlama.
- Best for: Quick summaries, factual lookups, when offline.
- **Command**: `synthesize(..., mode="local")`

### 2. Gemini Mode (High Quality)
- Uses **only** Gemini.
- Best for: Creative writing, complex reasoning, "Semantic Collision".
- **Command**: `synthesize(..., mode="gemini")`

### 3. Hybrid Mode (Default)
- **Step 1**: TinyLlama generates a "Factual Draft" based on retrieved evidence.
- **Step 2**: Gemini refines the draft, improving flow and tone, and adding strategic insights.
- **Benefit**: Reduces hallucination (grounded by local model) while maintaining high fluency.
- **Command**: `synthesize(..., mode="hybrid")`

---

## 💥 Semantic Collider

The Oracle is capable of **"Semantic Collision"**, a process where disparate pieces of evidence are forced together to generate novel insights.

**Prompt Strategy**:
> "Do not just summarize. **Collide** the evidence to generate sparks of new insights. **Fuse** seemingly unconnected concepts..."

This is primarily powered by the Strategic Expert (Gemini) due to the high abstraction capability required.

---

## 🛠️ Technical Implementation

### Local Optimization
The local model is optimized for consumer hardware (specifically Intel i9 CPUs):
- **Float32**: Used instead of FP16 for CPU performance.
- **Batch Size 1**: Optimized for real-time interaction.
- **Threads**: 8 threads for parallel processing.

### Fallback Mechanism
If the API Key is missing or the network is down, the system automatically degrades gracefully to **Local Mode**, ensuring the agent remains functional (albeit less eloquent).

---

**Last Updated**: 2025-12-04
**Version**: 1.0
**Status**: Active


---


# 📄 FILE: docs/core/field/README.md
================================================================================

# Campo Pré-Estrutural (`core/field/`)

> **Camada geométrica que opera no nível de potencial, não de estrutura.**

---

## Visão Geral

O Campo Pré-Estrutural é uma camada de processamento que trata o conhecimento como um **campo contínuo** em vez de estruturas discretas (grafos, tabelas). Ele unifica Learning e Reasoning através de geometria diferencial.

### Metáfora

| Sistema Tradicional | Campo Pré-Estrutural |
|---------------------|---------------------|
| Mapa de ruas | Campo gravitacional |
| Conexões fixas | Potencial que muda |
| "Onde as coisas estão" | "Para onde querem ir" |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    PreStructuralField                        │
│         (wrapper unificado com conexões VQ-VAE/Mycelial)    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │   Manifold  │───▶│   Metric    │───▶│    Field     │   │
│   │  (espaço)   │    │ (distâncias)│    │   F(x,t)     │   │
│   └─────────────┘    └─────────────┘    └──────────────┘   │
│          │                   │                  │           │
│          ▼                   ▼                  ▼           │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │  GeodesicFlow│   │CycleDynamics│───▶│ Cristalização│   │
│   │ (propagação)│    │(Exp→Cfg→Cmp)│    │   (→grafo)   │   │
│   └─────────────┘    └─────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes

### 1. DynamicManifold (`manifold.py`)

Variedade diferenciável com dimensão variável.

```python
from core.field import DynamicManifold

manifold = DynamicManifold()
point = manifold.embed(embedding_384d)
manifold.expand_dimension(n_dims=4)  # Cresce durante aprendizado
manifold.contract_dimension(n_dims=2)  # Comprime após consolidar
```

**Responsabilidades:**
- Projetar embeddings 384D em pontos da variedade
- Gerenciar dimensões dinâmicas (expansão/contração)
- Manter estrutura de vizinhança (KDTree)

---

### 2. RiemannianMetric (`metric.py`)

Métrica que deforma localmente quando conceitos são ativados.

```python
from core.field import RiemannianMetric

metric = RiemannianMetric(manifold)
metric.deform_at(point.coordinates, intensity=0.8)
distance = metric.distance(point_a, point_b)
curvature = metric.curvature_scalar_at(point)
```

**Responsabilidades:**
- Medir distâncias em espaço curvo
- Deformar espaço quando conceitos são "triggerados"
- Calcular curvatura (Ricci scalar)
- Computar símbolos de Christoffel para geodésicas

---

### 3. FreeEnergyField (`free_energy_field.py`)

Campo de energia livre F(x) = E(x) - T·S(x).

```python
from core.field import FreeEnergyField

field = FreeEnergyField(manifold, metric)
F = field.free_energy_at(point)
gradient = field.gradient_at(point)
attractors = field.get_state().attractors
```

**Fórmula:**
- **E(x)**: Energia interna (surpresa/prediction error)
- **S(x)**: Entropia (incerteza sobre transições)
- **T**: Temperatura (exploration vs exploitation)

**Responsabilidades:**
- Calcular energia livre em cada ponto
- Encontrar atratores (mínimos locais)
- Computar gradientes para dinâmica

---

### 4. GeodesicFlow (`geodesic_flow.py`)

Propagação via caminhos geodésicos (mais curtos em espaço curvo).

```python
from core.field import GeodesicFlow

flow = GeodesicFlow(manifold, metric)
path = flow.compute_geodesic(start, velocity)
shortest = flow.shortest_path(start, end)
```

**Responsabilidades:**
- Computar geodésicas (equação: d²x/dt² = -Γ v v)
- Propagar ativação seguindo geometria
- Gerar streamlines para visualização

---

### 5. CycleDynamics (`cycle_dynamics.py`)

O coração do sistema: ciclo Expansão → Configuração → Compressão.

```python
from core.field import CycleDynamics

cycle = CycleDynamics(manifold, metric, field, flow)
result = cycle.run_cycle(trigger_embedding)
```

**Fases:**
1. **Expansão**: Espaço cresce em dimensões novas
2. **Configuração**: Elementos se arranjam (annealing)
3. **Compressão**: Colapsa em estrutura densa

**Responsabilidades:**
- Orquestrar ciclo completo
- Atualizar regras de transição (meta-learning)
- Cristalizar campo em grafo

---

### 6. PreStructuralField (`pre_structural_field.py`)

Wrapper unificado que conecta com VQ-VAE e Mycelial.

```python
from core.field import PreStructuralField

# Inicializa
field = PreStructuralField()

# Conecta com resto do sistema
field.connect_vqvae(vqvae_model)
field.connect_mycelial(mycelial_reasoning)

# Usa
state = field.trigger(embedding)
states = field.propagate(steps=5)
graph = field.crystallize()

# Stats
print(field.stats())
```

---

## Uso Típico

```python
from core.field import PreStructuralField, PreStructuralConfig

# 1. Configurar
config = PreStructuralConfig(
    base_dim=384,
    temperature=1.0,
    max_expansion=32
)
field = PreStructuralField(config)

# 2. Trigger conceitos
for embedding in embeddings:
    state = field.trigger(embedding, intensity=0.8)
    print(f"F_mean = {state.mean_free_energy:.4f}")

# 3. Propagar dinâmica
states = field.propagate(steps=10)

# 4. Annealing (exploration → exploitation)
final_states = field.anneal(start_temp=2.0, end_temp=0.1, steps=50)

# 5. Cristalizar em grafo
graph = field.crystallize()
print(f"Atratores: {len(graph['nodes'])}")
print(f"Conexões: {len(graph['edges'])}")
```

---

## Integração com Alexandria

### Conexão com VQ-VAE
```python
field.connect_vqvae(vqvae_model)
# → Codebook vira anchor_points na variedade
```

### Conexão com Mycelial
```python
field.connect_mycelial(mycelial)
# → Cristalização alimenta grafo Hebbiano
```

### Conexão com VariationalFreeEnergy
```python
field.connect_variational_fe(vfe_module)
# → Sincroniza beliefs e componentes de F
```

---

## Arquivos

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `__init__.py` | 54 | Exports do módulo |
| `manifold.py` | 400 | DynamicManifold |
| `metric.py` | 436 | RiemannianMetric |
| `free_energy_field.py` | 500 | FreeEnergyField |
| `geodesic_flow.py` | 551 | GeodesicFlow |
| `cycle_dynamics.py` | 592 | CycleDynamics |
| `pre_structural_field.py` | 400 | PreStructuralField |

**Total: ~3,000 LOC**

---

### 7. DimensionalityReducer (`dim_reduction.py`)
Componente para otimização de hardware (redução 384d -> 32d).
- **Responsabilidade**: Projetar embeddings para espaço manifold de baixa dimensão
- **Algoritmo**: PCA incremental ou Random Projection
- **Motivação**: Cálculo geodésico eficiente em CPU

---

## Dependências

- numpy
- scipy (KDTree, integrate, ndimage)
- typing, dataclasses, logging

---

## Testes

```bash
python scripts/testing/test_field.py       # Teste rápido
python scripts/testing/test_field_simple.py # Teste com LanceDB real
```

---

## Matemática Base

### Energia Livre
```
F(x) = E(x) - T·S(x)

Onde:
- E = energia interna (surpresa)
- S = entropia
- T = temperatura
```

### Equação Geodésica
```
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0

Onde Γ = símbolos de Christoffel da métrica
```

### Deformação da Métrica
```
g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)

Onde:
- δ_ij = métrica plana (identidade)
- w_a = intensidade da deformação a
- c_a = centro da deformação a
- r = raio de influência
```


---


# 📄 FILE: docs/core/field/cycle_dynamics.md
================================================================================

# CycleDynamics (`core/field/cycle_dynamics.py`)

> Ciclo Expansão → Configuração → Compressão.

## Visão Geral

O `CycleDynamics` é o coração do Campo Pré-Estrutural. Implementa um ciclo de três fases que permite ao sistema descobrir estrutura:

```
      ┌─────────────────┐
      │    EXPANSÃO     │  ← Espaço cresce
      │  (mais dims)    │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │  CONFIGURAÇÃO   │  ← Elementos se arranjam
      │  (annealing)    │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │   COMPRESSÃO    │  ← Dimensões colapsam
      │  (menos dims)   │
      └────────┬────────┘
               │
               ▼
        estrutura emerge
```

## Fases

### 1. EXPANSÃO

- Dimensões são adicionadas
- Graus de liberdade aumentam
- Permite novas configurações

### 2. CONFIGURAÇÃO

- Simulated annealing
- Temperatura decresce gradualmente
- Elementos encontram posições de mínima energia

### 3. COMPRESSÃO

- Dimensões com baixa variância são removidas
- Informação é comprimida
- Estrutura densa emerge

## Uso

```python
from core.field import CycleDynamics, CycleConfig

config = CycleConfig(
    configuration_steps=50,
    expansion_threshold=0.7,    # F > 0.7 → expande
    compression_threshold=0.3,  # F < 0.3 → comprime
    max_expansion_dims=32
)

cycle = CycleDynamics(manifold, metric, field, flow, config)

# Rodar ciclo
trigger = np.random.randn(384)  # Embedding que inicia
result = cycle.run_cycle(trigger)

print(f"Fase: {result.phase}")
print(f"Ciclo #: {result.cycle_number}")
print(f"Dims adicionadas: {result.dimensions_added}")
print(f"Dims removidas: {result.dimensions_removed}")
print(f"ΔF: {result.free_energy_delta}")
```

## CycleState

```python
@dataclass
class CycleState:
    cycle_number: int
    phase: CyclePhase
    dimensions_added: int
    dimensions_removed: int
    free_energy_delta: float
    configuration_loss: float
    timestamp: str
```

## Cristalização

Após um ciclo, a estrutura pode ser "cristalizada" em grafo:

```python
graph = cycle.crystallize()
# {"nodes": [...], "edges": [...]}
```

## Auto-modificação

O ciclo pode modificar suas próprias regras:

```python
# Após N ciclos, regras de transição são atualizadas
cycle.update_rules(new_expansion_threshold=0.6)
```


---


# 📄 FILE: docs/core/field/free_energy_field.md
================================================================================

# FreeEnergyField (`core/field/free_energy_field.py`)

> Campo F(x) = E(x) - T·S(x) sobre a variedade.

## Visão Geral

O `FreeEnergyField` computa a energia livre em cada ponto da variedade. É o princípio unificador que determina para onde o sistema "quer ir".

## Fórmula

```
F(x) = E(x) - T·S(x)
```

| Componente | Símbolo | Descrição |
|------------|---------|-----------|
| Energia Livre | F | O que minimizamos |
| Energia Interna | E | Surpresa/prediction error |
| Entropia | S | Incerteza sobre transições |
| Temperatura | T | Trade-off exploration/exploitation |

## Classes

### FieldConfig

```python
@dataclass
class FieldConfig:
    temperature: float = 1.0        # T
    energy_scale: float = 1.0       # Escala de E
    entropy_scale: float = 1.0      # Escala de S
    gradient_step: float = 0.01     # Para gradientes numéricos
```

### FieldState

```python
@dataclass
class FieldState:
    timestamp: str
    mean_free_energy: float
    min_free_energy: float
    max_free_energy: float
    num_attractors: int
    attractors: List[np.ndarray]
    gradient_field: np.ndarray
```

## Uso

```python
from core.field import FreeEnergyField, FieldConfig

field = FreeEnergyField(manifold, metric)

# Energia em um ponto
F = field.free_energy_at(point.coordinates)
E = field.energy_at(point.coordinates)
S = field.entropy_at(point.coordinates)

# Gradiente (direção de descida)
grad = field.gradient_at(point.coordinates)

# Estado do campo inteiro
state = field.compute_field()
print(f"F_mean = {state.mean_free_energy}")
print(f"Atratores = {state.num_attractors}")

# Ajustar temperatura
field.set_temperature(0.5)  # Mais exploitation
field.set_temperature(2.0)  # Mais exploration

# Stats
stats = field.stats()
```

## Temperatura

| T | Comportamento |
|---|---------------|
| T → 0 | Sistema vai para mínimos de energia (greedy) |
| T = 1 | Balanceado |
| T → ∞ | Sistema ignora energia, maximiza entropia (random) |

## Atratores

Atratores são **mínimos locais** de F:

```python
attractors = field.get_state().attractors
# Lista de coordenadas onde ∇F ≈ 0 e F é mínimo local
```


---


# 📄 FILE: docs/core/field/geodesic_flow.md
================================================================================

# GeodesicFlow (`core/field/geodesic_flow.py`)

> Propagação via caminhos geodésicos.

## Visão Geral

O `GeodesicFlow` computa geodésicas — os caminhos mais curtos em espaço curvo. São análogos a "linhas retas" em espaço plano, mas seguem a curvatura.

## Equação Geodésica

```
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

Onde `Γ` são os símbolos de Christoffel da métrica.

## Uso

```python
from core.field import GeodesicFlow, GeodesicConfig

config = GeodesicConfig(
    max_steps=100,
    step_size=0.01,
    use_scipy_integrator=True  # Mais preciso, mais lento
)
flow = GeodesicFlow(manifold, metric, config)

# Geodésica a partir de ponto e velocidade
start = point.coordinates
velocity = np.random.randn(dim) * 0.1
path = flow.compute_geodesic(start, velocity, max_steps=50)

print(f"Passos: {path.n_steps}")
print(f"Comprimento: {path.length}")
print(f"Convergiu: {path.converged}")

# Caminho mais curto entre dois pontos
shortest = flow.shortest_path(start, end)
```

## GeodesicPath

```python
@dataclass
class GeodesicPath:
    points: np.ndarray      # [n_steps, dim]
    velocities: np.ndarray  # [n_steps, dim]
    length: float
    n_steps: int
    converged: bool
```

## Casos de Uso

1. **Propagação de ativação**: Conceito ativado propaga ao longo de geodésicas
2. **Conexão de clusters**: Encontrar caminho natural entre conceitos distantes
3. **Visualização**: Mostrar "linhas de fluxo" no espaço de conhecimento


---


# 📄 FILE: docs/core/field/manifold.md
================================================================================

# DynamicManifold (`core/field/manifold.py`)

> Variedade diferenciável com dimensão variável.

## Visão Geral

O `DynamicManifold` representa o espaço onde o conhecimento vive. Não é um grafo discreto — é um **espaço contínuo** que pode expandir e contrair suas dimensões.

## Conceitos

| Conceito | Descrição |
|----------|-----------|
| Pontos | Conceitos/embeddings projetados na variedade |
| Códigos VQ-VAE | Coordenadas discretas (âncoras) |
| Dimensão | Pode crescer (expansão) e encolher (compressão) |
| Topologia | Emerge da distribuição de pontos |

## Classes

### ManifoldConfig

```python
@dataclass
class ManifoldConfig:
    base_dim: int = 384           # Dimensão base (embedding)
    num_heads: int = 4            # Heads do VQ-VAE
    codebook_size: int = 256      # Códigos por head
    max_expansion: int = 128      # Máximo de dimensões extras
    sparsity_threshold: float = 0.01
    neighborhood_k: int = 16
```

### ManifoldPoint

```python
@dataclass
class ManifoldPoint:
    coordinates: np.ndarray      # Coordenadas contínuas [dim]
    discrete_codes: np.ndarray   # Códigos VQ-VAE [4]
    activation: float = 0.0      # Nível de ativação
    metadata: Dict[str, Any]
```

## Uso

```python
from core.field import DynamicManifold, ManifoldConfig

# Criar variedade
config = ManifoldConfig(base_dim=384)
manifold = DynamicManifold(config)

# Projetar embedding
embedding = np.random.randn(384)
point = manifold.embed(embedding)

# Adicionar à variedade
manifold.add_point("concept_1", point)

# Ativar (trigger)
manifold.activate_point("concept_1", intensity=0.8)

# Buscar vizinhos
neighbors = manifold.get_neighbors(point, k=5)
# [(point_id, distance), ...]

# Expandir dimensões
manifold.expand_dimension(n_dims=4)
print(manifold.current_dim)  # 388

# Contrair
manifold.contract_dimension(n_dims=2)
print(manifold.current_dim)  # 386

# Estatísticas
stats = manifold.stats()
# {
#     "num_points": 100,
#     "current_dim": 386,
#     "active_points": 15,
#     "mean_activation": 0.23
# }
```

## Integração com VQ-VAE

```python
# Definir pontos âncora do codebook
manifold.set_anchor_points(vqvae.get_codebook())

# Criar ponto a partir de códigos
point = manifold.from_vqvae_codes([12, 45, 200, 78])
```

## Serialização

```python
# Salvar
data = manifold.to_dict()

# Carregar
manifold = DynamicManifold.from_dict(data)
```


---


# 📄 FILE: docs/core/field/metric.md
================================================================================

# RiemannianMetric (`core/field/metric.py`)

> Métrica que deforma localmente com ativação.

## Visão Geral

A `RiemannianMetric` mede distâncias em um espaço curvo. Quando conceitos são ativados, eles **deformam** o espaço ao redor, criando "poços" de atração.

## Matemática

### Métrica Deformada

```
g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)
```

Onde:
- `δ_ij` = métrica plana (identidade)
- `w_a` = intensidade da deformação a
- `c_a` = centro da deformação a
- `r` = raio de influência

### Curvatura

A curvatura escalar R indica "quão curvado" está o espaço:
- R = 0: espaço plano
- R > 0: espaço curvado positivamente (esfera)
- R < 0: espaço curvado negativamente (sela)

## Uso

```python
from core.field import RiemannianMetric, MetricConfig, DynamicManifold

# Criar
manifold = DynamicManifold()
config = MetricConfig(
    deformation_radius=0.3,
    deformation_strength=0.5
)
metric = RiemannianMetric(manifold, config)

# Deformar em um ponto
point = manifold.points["concept_1"]
metric.deform_at(point.coordinates, intensity=0.8)

# Medir distância (não-Euclidiana)
dist = metric.distance(point_a, point_b)

# Curvatura local
R = metric.curvature_scalar_at(point.coordinates)

# Relaxar deformações
metric.relax(rate=0.1)  # Decai 10%

# Stats
print(metric.stats())
# {"deformations": 5, "total_curvature": 0.23}
```

## Símbolos de Christoffel

Para geodésicas, calculamos os símbolos de Christoffel:

```python
Γ = metric.christoffel_at(point.coordinates)
# Tensor [dim, dim, dim]
```

Esses símbolos definem como vetores mudam ao se mover no espaço curvo.

## Integração

A métrica é usada pelo:
- `GeodesicFlow`: para propagar no espaço curvo
- `FreeEnergyField`: para calcular gradientes
- `CycleDynamics`: para detectar atratores


---


# 📄 FILE: docs/core/integrations/abduction_compositional_integration.md
================================================================================

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


---


# 📄 FILE: docs/core/integrations/agents_compositional_integration.md
================================================================================

# 🤖 Agents Compositional Integration

**Module**: `core/integrations/agents_compositional_integration.py`
**Lines of Code**: ~891
**Purpose**: Connect all agents (Action, Bridge, Critic, Oracle) to compositional reasoning on the curved manifold.

---

## 🎯 Overview

This integration enhances each agent with **geometric awareness**:

| Agent | Enhancement |
|-------|-------------|
| **ActionAgent** | Selects actions by evaluating geodesic paths to goals |
| **BridgeAgent** | Translates between representations following geodesics |
| **CriticAgent** | Scores outputs based on path energy and coherence |
| **Oracle** | Synthesizes responses by walking the manifold |

---

## 📊 Core Classes

### `GeometricActionResult`
```python
@dataclass
class GeometricActionResult:
    action_type: str
    target_state: np.ndarray
    geodesic_path: np.ndarray
    expected_reward: float
    path_energy: float
    confidence: float
```

### `GeometricTranslation`
```python
@dataclass
class GeometricTranslation:
    source: np.ndarray
    target: np.ndarray
    intermediate_path: np.ndarray
    translation_fidelity: float
    structure_preserved: float
```

### `GeometricCritique`
```python
@dataclass
class GeometricCritique:
    overall_score: float
    energy_score: float
    complexity_score: float
    coherence_score: float
    suggestions: List[str]
```

---

## 🎯 Agent Classes

### `GeometricActionAgent`
Selects actions via geodesic path evaluation.

### `GeometricBridgeAgent`
Translates between representations by following geodesics.

### `GeometricCriticAgent`
Critiques outputs based on geometric properties.

### `GeometricOracle`
Synthesizes responses by following compositional paths.

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Core geometric computations.
- **ActionAgent**, **BridgeAgent**, **CriticAgent**, **Oracle**: Base agent logic.
- **CompositionalReasoner**: Path composition.

---

**Last Updated**: 2025-12-11
**Status**: Production


---


# 📄 FILE: docs/core/integrations/alexandria_unified.md
================================================================================

# 🧬 Alexandria Unified Integration

**Module**: `core/integrations/alexandria_unified.py`
**Lines of Code**: ~718
**Purpose**: Master integration layer that unifies all subsystems into a single coherent cognitive cycle.

---

## 🎯 Overview

The **AlexandriaCore** class is the central orchestrator of the Alexandria system. It wraps all specialized integrations (Geodesic, Nemesis, Abduction, Agents, Loop) into a unified interface, providing a single entry point for executing complete cognitive cycles.

### Why It Exists
Rather than manually wiring the Bridge module, the Field, the Compositional Reasoner, and Learning layers in every script, this module encapsulates the complexity. You instantiate one `AlexandriaCore` and call `cognitive_cycle()`.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "AlexandriaCore Facade"
        direction TB
        Core[AlexandriaCore] --> Bridge[GeodesicBridgeIntegration]
        Core --> Nemesis[NemesisBridgeIntegration]
        Core --> Abduction[AbductionCompositionalIntegration]
        Core --> Agents[AgentsCompositionalIntegration]
        Core --> Loop[LoopCompositionalIntegration]
    end
    
    subgraph "Lower Layers"
        Bridge --> Manifold[FieldManifold]
        Nemesis --> FreeEnergy[Free Energy Computation]
        Abduction --> VQ[VQ-VAE Codes]
    end
    
    User[User Code] --> Core
    Core --> Result[CognitiveCycleResult]
```

---

## 📊 Core Classes

### `AlexandriaConfig`
Central configuration dataclass for all subsystem parameters.
```python
@dataclass
class AlexandriaConfig:
    bridge_pull_strength: float = 0.5
    nemesis_risk_weight: float = 1.0
    abduction_max_hypotheses: int = 5
    loop_max_iterations: int = 100
    loop_exploration_rate: float = 0.1
```

### `CognitiveCycleResult`
Output of one complete think-act-learn cycle.
```python
@dataclass
class CognitiveCycleResult:
    perception: Dict[str, Any]
    reasoning: Dict[str, Any]
    gaps_detected: List[Dict]
    hypotheses_generated: List[Dict]
    action_selected: Dict[str, Any]
    learning_metrics: Dict[str, float]
    free_energy: float
    iteration: int
    duration_ms: float
```

---

## 🎯 Use Cases

### 1. Run a Single Cognitive Cycle
```python
from core.integrations.alexandria_unified import AlexandriaCore
import numpy as np

core = AlexandriaCore.from_vqvae(vqvae_model)
observation = np.random.randn(384)

result = core.cognitive_cycle(observation)
print(f"Free Energy: {result.free_energy:.4f}")
print(f"Gaps: {len(result.gaps_detected)}, Hypotheses: {len(result.hypotheses_generated)}")
```

### 2. Health Check
```python
status = core.health_check()
# Returns dict with component availability
```

---

## 🔗 Dependencies

- **Requires**: `GeodesicBridgeIntegration`, `NemesisBridgeIntegration`, `AbductionCompositionalIntegration`, `AgentsCompositionalIntegration`, `LoopCompositionalIntegration`.
- **Used By**: `scripts/demos/demo_full_system.py`, `interface/pages/4_🔮_Abduction.py`.

---

**Last Updated**: 2025-12-11
**Version**: 1.0
**Status**: Production


---


# 📄 FILE: docs/core/integrations/geodesic_bridge_integration.md
================================================================================

# 🛤️ Geodesic Bridge Integration

**Module**: `core/integrations/geodesic_bridge_integration.py`
**Lines of Code**: ~656
**Purpose**: Connect the geodesic flow engine to the curved manifold (VQ-VAE Bridge), enabling semantic path computation.

---

## 🎯 Overview

This integration allows Alexandria to compute **semantic paths** (geodesics) between concepts in the curved latent space defined by the VQ-VAE manifold. Instead of straight-line (Euclidean) paths, concepts are connected via **geodesics** that follow the learned metric.

### Key Capabilities
1. **Semantic Path Finding**: Compute the "shortest" path between two embeddings respecting curvature.
2. **Activation Propagation**: Spread activation from a concept radially through the manifold.
3. **Geodesic Fields**: Explore all directions from a point to map local geometry.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Inputs"
        Start[Start Embedding]
        End[End Embedding]
    end
    
    subgraph "GeodesicBridgeIntegration"
        Bridge[VQVAEManifoldBridge]
        GF[GeodesicFlow Engine]
        Cache[Path Cache]
    end
    
    Start --> Bridge
    End --> Bridge
    Bridge --> GF
    GF --> Path[SemanticPath]
    Path --> Cache
```

---

## 📊 Core Classes

### `GeodesicBridgeConfig`
Configuration dataclass for tuning geodesic computation.

### `SemanticPath`
Result of a geodesic computation:
```python
@dataclass
class SemanticPath:
    start: np.ndarray
    end: np.ndarray
    points: np.ndarray          # Intermediate waypoints
    geodesic_length: float      # Path length in manifold
    euclidean_length: float     # Straight-line distance
    curvature_integral: float   # Total curvature traversed
    attractors_visited: List[int]
    converged: bool
```

### `ActivationMap`
Result of spreading activation:
```python
@dataclass
class ActivationMap:
    source: np.ndarray
    activations: Dict[int, float]  # Concept ID -> activation level
    total_energy: float
    n_activated: int
```

---

## 🎯 Key Methods

| Method | Description |
|--------|-------------|
| `semantic_path(start, end)` | Computes geodesic between two embeddings |
| `propagate_activation(source, radius)` | Spreads activation from a point |
| `geodesic_field(center, n_directions)` | Maps local geometry |
| `distance(a, b, geodesic=True)` | Computes distance (geodesic or Euclidean) |

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Provides the metric tensor and curvature.
- **GeodesicFlow**: Core geodesic ODE solver.

**Used By**: `AlexandriaCore`, `NemesisBridgeIntegration`.

---

**Last Updated**: 2025-12-11
**Version**: 1.0
**Status**: Production


---


# 📄 FILE: docs/core/integrations/learning_field_integration.md
================================================================================

# 📚 Learning Field Integration

**Module**: `core/integrations/learning_field_integration.py`
**Lines of Code**: ~858
**Purpose**: Connect the Learning Stack (Predictive Coding, Active Inference, Meta-Hebbian) to the Pre-Structural Field.

---

## 🎯 Overview

This integration unifies the three main learning paradigms with the curved manifold:

1. **Predictive Coding**: Prediction errors are computed as geodesic distances.
2. **Active Inference**: Action planning uses geodesic EFE.
3. **Meta-Hebbian**: Learning rates are modulated by local curvature.

### Key Insight
By computing errors and plans geodesically, the system respects the learned topology of knowledge. Errors "in the direction of" strong conceptual connections are less surprising than orthogonal deviations.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Learning Stack"
        PC[Predictive Coding]
        AI[Active Inference]
        MH[Meta-Hebbian]
    end
    
    subgraph "Field"
        Bridge[VQVAEManifoldBridge]
        Metric[Riemannian Metric]
        Geodesic[Geodesic Flow]
    end
    
    PC -->|Geodesic Error| Bridge
    AI -->|Geodesic EFE| Bridge
    MH -->|Curvature Modulation| Metric
    
    Bridge --> Geodesic
    Metric --> Geodesic
```

---

## 📊 Core Classes

### `GeometricPrediction`
```python
@dataclass
class GeometricPrediction:
    predicted: np.ndarray
    observed: np.ndarray
    error: np.ndarray
    geodesic_error: float       # Distance in manifold
    euclidean_error: float      # Straight-line error
    curvature_at_prediction: float
    precision: float
```

### `GeometricPredictiveCoding`
PC layer that uses geodesic distance for error computation.

### `GeometricActiveInference`
AI agent that plans in the curved manifold.

### `GeometricMetaHebbian`
Meta-learning layer where learning rates depend on local curvature.

### `LearningFieldIntegration`
Master class that coordinates all three geometric learning modules.

---

## 🎯 Key Methods

| Method | Description |
|--------|-------------|
| `full_cycle(observation, context)` | Runs PC→encode→AI→plan→MH→adapt |
| `encode(observation)` | Geodesic predictive coding encoding |
| `plan(current, goal)` | Geodesic action planning |
| `adapt(prediction)` | Curvature-modulated learning |

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Geodesic and metric computations.
- **PredictiveCodingNetwork**, **ActiveInferenceAgent**, **MetaHebbianOptimizer**: Base learning modules.

**Used By**: `AlexandriaCore`, `LoopCompositionalIntegration`.

---

**Last Updated**: 2025-12-11
**Version**: 1.0
**Status**: Production


---


# 📄 FILE: docs/core/integrations/loop_compositional_integration.md
================================================================================

# 🔄 Loop Compositional Integration

**Module**: `core/integrations/loop_compositional_integration.py`
**Lines of Code**: ~929
**Purpose**: Connect the Self-Feeding Loop to compositional reasoning, closing the autonomous learning cycle.

---

## 🎯 Overview

This integration closes the **autonomous learning cycle** where the system:
1. **Perceive**: Encodes observations into the manifold.
2. **Reason**: Generates hypotheses as geodesic paths.
3. **Act**: Executes actions to test hypotheses.
4. **Learn**: Updates the metric based on feedback.

### Key Capability
The loop is now **geometry-aware**: feedback doesn't just update weights—it **deforms the manifold**, making successful paths easier to traverse in the future.

---

## 📊 Core Classes

### `LoopPhase`
```python
class LoopPhase(Enum):
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    LEARN = "learn"
```

### `LoopState`
```python
@dataclass
class LoopState:
    iteration: int
    phase: LoopPhase
    current_point: np.ndarray
    belief: np.ndarray
    free_energy: float
    prediction_error: float
    cumulative_reward: float
```

### `CycleResult`
```python
@dataclass
class CycleResult:
    trajectory: List[np.ndarray]
    actions_taken: List[Dict]
    total_iterations: int
    final_free_energy: float
    energy_reduction: float
    hypotheses_generated: int
    hypotheses_validated: int
    metric_deformations: int
    duration_seconds: float
```

### `GeodesicFeedback`
Feedback that propagates along geodesics from the source point.

---

## 🎯 Key Methods

| Method | Description |
|--------|-------------|
| `autonomous_cycle(observation, goal)` | Runs full perceive→reason→act→learn loop until convergence |
| `_perceive_phase()` | Updates belief from observation |
| `_reason_phase()` | Generates hypotheses via abduction |
| `_act_phase(hypothesis, goal)` | Executes action |
| `_learn_phase(action, reward)` | Updates metric and validates hypotheses |
| `propagate_feedback(point, value)` | Spreads feedback geodesically |

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Metric updates and geodesics.
- **SelfFeedingLoop**: Base loop logic.
- **NemesisIntegration**: Active Inference for action selection.
- **AbductionCompositionalIntegration**: Hypothesis generation.

---

**Last Updated**: 2025-12-11
**Status**: Production


---


# 📄 FILE: docs/core/integrations/nemesis_bridge_integration.md
================================================================================

# 🧠 Nemesis Bridge Integration

**Module**: `core/integrations/nemesis_bridge_integration.py`
**Lines of Code**: ~724
**Purpose**: Connect the Active Inference engine (Nemesis) to the curved manifold for geometry-aware action selection.

---

## 🎯 Overview

The **NemesisBridgeIntegration** extends Active Inference to operate on a curved semantic space. The Expected Free Energy (EFE) is computed using **geodesic distance** instead of Euclidean, making the agent aware of the learned knowledge topology.

### Key Capabilities
1. **Geometric Action Selection**: Choose actions that minimize EFE computed geodesically.
2. **Curvature-Aware Planning**: Bonus for exploring high-curvature regions (interesting areas).
3. **Belief Updating**: Update generative model based on geometric prediction errors.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Inputs"
        Gaps[Knowledge Gaps]
        Hypo[Hypotheses]
        State[Current State]
    end
    
    subgraph "NemesisBridgeIntegration"
        Bridge[VQVAEManifoldBridge]
        Nemesis[Active Inference Core]
        Comp[Compositional Reasoner]
    end
    
    Gaps --> Nemesis
    Hypo --> Nemesis
    State --> Bridge
    Bridge --> Nemesis
    Comp --> Nemesis
    Nemesis --> Action[GeometricAction]
```

---

## 📊 Core Classes

### `GeometricEFE`
Expected Free Energy with geometric components:
```python
@dataclass
class GeometricEFE:
    total: float
    risk: float                 # Distance to preferred state
    ambiguity: float            # Uncertainty along path
    curvature_term: float       # Bonus for curved regions
    field_energy: float         # Free energy at target
    geodesic_length: float
    attractors_visited: int
```

### `GeometricAction`
Action enriched with geometric information:
```python
@dataclass
class GeometricAction:
    action_type: str
    target: str
    parameters: Dict[str, Any]
    geometric_efe: GeometricEFE
    geodesic_path: Optional[np.ndarray]
    epistemic_value: float
    pragmatic_value: float
    geometric_value: float
```

---

## 🎯 Key Methods

| Method | Description |
|--------|-------------|
| `select_action_geometric(gaps, hypotheses, state)` | Selects best action via geometric EFE |
| `update_beliefs_geometric(observation, action, reward)` | Updates model post-action |
| `_compute_geometric_efe(state, action)` | Computes full EFE with all terms |

---

## 🔗 Dependencies
- **VQVAEManifoldBridge**: Metric and field computations.
- **NemesisIntegration**: Base Active Inference logic.
- **CompositionalReasoner**: Optional, for compositional action generation.

**Used By**: `AlexandriaCore.cognitive_cycle()`.

---

**Last Updated**: 2025-12-11
**Version**: 1.0
**Status**: Production


---


# 📄 FILE: docs/core/learning/NEMESIS_MANUAL.md
================================================================================

# 💀 Cognitive Nemesis: User Manual

**"Efficiency is the ultimate sophistication."**

This manual documents the **Cognitive Nemesis** architecture, a resource-efficient, multi-agent cognitive layer atop the Alexandria system. It is designed to run on consumer hardware (Lite Mode) whilst delivering "operational excellence" through specialized reasoning profiles.

---

## 🏗️ Architecture

The Nemesis architecture introduces three key components:

1.  **Multi-Agent Orchestrator**: Manages specialized agents accessing a shared `MycelialReasoning` graph.
2.  **Reasoning Profiles**: Distinct cognitive personalities (Scout, Judge, Weaver).
3.  **Lite Mode**: Aggressive resource management (Pruning) for local deployment.

```mermaid
graph TD
    User -->|Text| Orch[Multi-Agent Orchestrator]
    
    subgraph SharedMind [Shared Mycelial Memory]
        Graph[Sparse Graph]
        Lance[Vector Store]
    end
    
    subgraph Agents [Reasoning Profiles]
        Scout[🕵️ The Scout]
        Judge[⚖️ The Judge]
        Weaver[🕸️ The Weaver]
    end
    
    Orch --> Scout & Judge & Weaver
    Scout & Judge & Weaver <-->|Read/Write| SharedMind
    
    subgraph Governance [Free Energy Principle]
        Orch -->|Min EFE| FEP[Free Energy Minimization]
    end
```

---

## 🎭 The Three Faces (Agents)

Each agent uses **Active Inference** but with tuned parameters (Risk, Ambiguity, Novelty).

| Profile | Archetype | Function | Personality |
| :--- | :--- | :--- | :--- |
| **The Scout** | Explorer | Finds new information & novelty. | Fast, High Temperature, High Novelty Bonus. |
| **The Judge** | Verifier | Validates truth & prunes weak links. | Slow, Critical, Low Risk Tolerance. |
| **The Weaver** | Architect | Connects disparate concepts. | Balanced, Focus on Graph Density. |

### Configuration (`core/learning/profiles.py`)
Profiles are defined as dataclasses and can be hot-swapped or extended.

---

## ⚡ Lite Mode (Local Optimization)

Designed for 8GB-16GB RAM machines.

### Features
*   **Graph Pruning**: Automatically removes "orphaned" or weak nodes when graph size > 50k nodes.
*   **Vector Bypass**: Can skip loading heavy embedding models if vectors exist in LanceDB.
*   **Lazy Loading**: Agents only spin up when invoked.

### Usage
To activate Lite Mode:
```python
from core.learning.integration_layer import IntegrationConfig, SystemProfile

config = IntegrationConfig(profile=SystemProfile.LITE)
# Limits: 8GB RAM, 50k Graph Nodes, 3 PC Iterations
```

---

## 🛠️ Integration with Alexandria

The Nemesis layer sits between the **Ingestion** (LanceDB) and the **User**.

1.  **Ingestion**: Papers enter LanceDB (155k+ chunks).
2.  **Weaving**: *The Weaver* runs in background, connecting these chunks in the Mycelial Graph.
3.  **Querying**: *The Scout* explores the graph to answer user queries.
4.  **Maintenance**: *The Judge* periodically cleans the graph (Lite Mode maintenance).

---

## 📄 Scripts

*   `scripts/debug/verify_nemesis.py`: Verify connectivity and agent spawning.
*   `scripts/debug/test_nemesis_integration.py`: Test agents on real LanceDB data.
*   `scripts/train_mycelial.py`: Train the graph (Legacy, being updated for Nemesis).

---

**Last Updated**: 2025-12-05
**Status**: Operational


---


# 📄 FILE: docs/core/learning/README.md
================================================================================

# 🧠 Learning - Documentação dos Módulos

**Sistema de aprendizado cognitivo do Alexandria**

---

## 🗂️ Estrutura

```
docs/core/learning/
├── active_inference.md     → Agente autônomo de exploração
├── free_energy.md          → Princípio de energia livre (unificador)
├── predictive_coding.md    → Inferência hierárquica preditiva
├── meta_hebbian.md         → Meta-aprendizado via plasticidade
├── integration_layer.md    → Integração entre módulos
├── profiles.md             → Personalidades cognitivas
└── NEMESIS_MANUAL.md       → Manual do sistema NEMESIS
```

---

## 📊 Módulos por Hierarquia Conceitual

```mermaid
graph TB
    subgraph "Topo: Unificação"
        FE[Free Energy]
    end
    
    subgraph "Ação"
        AI[Active Inference]
    end
    
    subgraph "Percepção"
        PC[Predictive Coding]
    end
    
    subgraph "Plasticidade"
        MH[Meta-Hebbian]
    end
    
    subgraph "Integração"
        IL[Integration Layer]
        PR[Profiles]
    end
    
    FE --> AI
    FE --> PC
    AI --> MH
    PC --> MH
    MH --> IL
    PR --> AI
    
    style FE fill:#9C27B0,color:#fff
    style AI fill:#E91E63,color:#fff
    style PC fill:#FF9800,color:#fff
    style MH fill:#4CAF50,color:#fff
```

---

## 📈 Módulos

| Módulo | Arquivo | LOC | Descrição |
|--------|---------|-----|-----------|
| Active Inference | [active_inference.md](active_inference.md) | 1486 | Agente de exploração epistêmica autônoma |
| Free Energy | [free_energy.md](free_energy.md) | 1257 | Princípio unificador (topo da hierarquia) |
| Predictive Coding | [predictive_coding.md](predictive_coding.md) | 981 | Inferência por minimização de erro |
| Meta-Hebbian | [meta_hebbian.md](meta_hebbian.md) | 784 | Regras de plasticidade evoluídas |
| Integration Layer | [integration_layer.md](integration_layer.md) | 1223 | Ponto de conexão entre módulos |
| Profiles | [profiles.md](profiles.md) | 102 | Scout, Judge, Weaver personalities |
| NEMESIS Manual | [NEMESIS_MANUAL.md](NEMESIS_MANUAL.md) | - | Manual completo do sistema |

---

## 🔗 Fluxo de Dados

```
Observação → Predictive Coding → Beliefs
                    ↓
            Free Energy (compute F)
                    ↓
            Active Inference (plan action)
                    ↓
            Meta-Hebbian (update rules)
                    ↓
            Ação Executada → Nova Observação
```

---

**Última atualização**: 2025-12-07


---


# 📄 FILE: docs/core/learning/active_inference.md
================================================================================

# 🧠 Active Inference

**Module**: `core/learning/active_inference.py`  
**Lines of Code**: 1486  
**Purpose**: Agente de Active Inference para seleção de ações epistêmicas

---

## 🎯 Overview

O Active Inference implementa um **agente cognitivo autônomo** baseado no framework de Karl Friston. O agente mantém crenças (beliefs) sobre o estado do conhecimento, tem preferências sobre estados desejados, e seleciona ações que minimizam a energia livre esperada (Expected Free Energy).

Este é o módulo que dá **autonomia** ao Alexandria - permite que o sistema decida proativamente o que explorar, quais gaps investigar, e como expandir o conhecimento autonomamente.

### Conceito Chave: Expected Free Energy

```
G(π) = Risk + Ambiguity
     = D_KL[Q(o|π)||P(o)] + E_Q[H(o|s,π)]

Risk: Quão longe das preferências
Ambiguity: Incerteza sobre resultados
```

O agente escolhe ações que **minimizam G** - ou seja, que levam a estados preferidos (baixo risk) e que reduzem incerteza (baixa ambiguity).

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Core["Active Inference Agent"]
        B[Beliefs Q(s)]
        P[Preferences P(o)]
        GM[Generative Model]
    end
    
    subgraph Actions["Tipos de Ação"]
        A1[QUERY_SEARCH]
        A2[EXPLORE_CLUSTER]
        A3[FOLLOW_CONNECTION]
        A4[DEEPEN_TOPIC]
        A5[BRIDGE_CONCEPTS]
        A6[FILL_GAP]
    end
    
    subgraph Planning["Seleção de Ação"]
        EFE[Expected Free Energy G]
        PI[Policy π]
    end
    
    B --> EFE
    P --> EFE
    GM --> EFE
    EFE --> PI
    PI --> Actions
    
    style Core fill:#E91E63,color:#fff
    style Planning fill:#2196F3,color:#fff
```

---

## 🔍 Modelo Generativo

O `GenerativeModel` encapsula a dinâmica do mundo:

```python
# P(s_t+1 | s_t, a_t): Transição de estado dado ação
def predict_next_state(self, current_state, action):
    transition_matrix = self.transition_models[action.action_type]
    mean = transition_matrix @ current_state
    return mean, precision

# P(o_t | s_t): Observação esperada dado estado
def predict_observation(self, state):
    return self.observation_model @ state
```

### Aprendizado do Modelo

```python
# Δ A = lr * (s_t+1 - A @ s_t) @ s_t.T
def update_from_experience(self, prev_state, action, next_state):
    prediction = self.transition_models[action.action_type] @ prev_state
    error = next_state - prediction
    self.transition_models[action.action_type] += lr * np.outer(error, prev_state)
```

---

## 🔍 Seleção de Ação

### Expected Free Energy

```python
def _compute_expected_free_energy(self, action, current_state, horizon):
    """
    G(π) = Σ_t [Risk_t + Ambiguity_t]
    """
    total_G = 0
    state = current_state
    
    for t in range(horizon):
        # Prediz próximo estado
        next_state, precision = self.generative_model.predict_next_state(state, action)
        
        # Risk: distância das preferências
        risk = self._compute_risk(next_state)
        
        # Ambiguity: incerteza da predição
        ambiguity = self._compute_ambiguity(precision)
        
        total_G += risk + ambiguity
        state = next_state
    
    return total_G
```

### Planejamento

```python
def plan(self, horizon=5, context=None):
    """Seleciona melhor ação via Monte Carlo"""
    
    candidate_actions = self._generate_candidate_actions(context)
    
    for action in candidate_actions:
        G = self._compute_expected_free_energy(action, self.current_state, horizon)
        action.expected_information_gain = -G  # Menor G = maior ganho
    
    # Softmax sobre -G para seleção probabilística
    return self._softmax_select(candidate_actions)
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    AI[Active Inference] --> MH[Meta-Hebbian]
    AI --> PC[Predictive Coding]
    AI --> FE[Free Energy]
    
    style AI fill:#E91E63,color:#fff
```

**1. Meta-Hebbian** (`meta_hebbian.py`)
- **Purpose**: Plasticidade adaptativa
- **Call**: `meta_hebbian.compute_weight_update()`
- **When**: Após ação bem-sucedida

**2. Free Energy** (`free_energy.py`)
- **Purpose**: Métricas de minimização
- **Call**: `free_energy.compute()`
- **When**: Avaliação de estados

### **Used By**:

```mermaid
graph LR
    IL[Integration Layer] --> AI[Active Inference]
    AA[Action Agent] --> AI
    
    style AI fill:#E91E63,color:#fff
```

---

## 📊 Core Classes

### ActionType

```python
class ActionType(Enum):
    QUERY_SEARCH = auto()       # Buscar documentos
    EXPLORE_CLUSTER = auto()    # Explorar cluster de conceitos
    FOLLOW_CONNECTION = auto()  # Seguir conexão existente
    DEEPEN_TOPIC = auto()       # Aprofundar em tópico
    BRIDGE_CONCEPTS = auto()    # Conectar conceitos distantes
    FILL_GAP = auto()           # Preencher gap de conhecimento
    CONSOLIDATE = auto()        # Consolidar aprendizado
    REST = auto()               # Não fazer nada
```

### Action

```python
@dataclass
class Action:
    action_type: ActionType
    target: str                          # Alvo da ação
    parameters: Dict[str, Any]
    expected_information_gain: float     # Ganho esperado
    expected_risk: float
    priority: float
```

### Belief

```python
@dataclass
class Belief:
    concept_id: str
    mean: np.ndarray          # Estado estimado
    precision: np.ndarray     # Inverso da variância
    last_updated: int
    observation_count: int
    
    @property
    def uncertainty(self):
        return 1.0 / np.mean(self.precision)
```

### Preference

```python
@dataclass
class Preference:
    topic_weights: Dict[str, float]  # Peso por tópico
    novelty_preference: float = 0.5   # 0=conservador, 1=explorador
    depth_preference: float = 0.5     # 0=amplo, 1=profundo
    connection_preference: float = 0.5
```

---

## 🎯 Use Cases

### 1. Exploração Autônoma

```python
from core.learning.active_inference import ActiveInferenceAgent, ActiveInferenceConfig

config = ActiveInferenceConfig(
    planning_horizon=5,
    novelty_bonus=0.3
)
agent = ActiveInferenceAgent(config)

# Agente decide próxima ação
action = agent.plan(context={"current_topic": "neural networks"})
print(f"Ação escolhida: {action.action_type} → {action.target}")
```

### 2. Detecção de Gaps

```python
# Identifica gaps no conhecimento
gaps = agent.detect_knowledge_gaps()

for gap in gaps[:5]:
    print(f"Gap: {gap.description} (priority: {gap.priority:.2f})")
```

### 3. Ciclo Completo

```python
# Loop autônomo de aprendizado
while agent.should_continue():
    action = agent.plan()
    observation = execute_action(action)  # Externo
    agent.update_beliefs(observation)
    agent.learn_from_experience(action, observation)
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **plan()** | ~50ms | 5-step horizon |
| **update_beliefs()** | ~5ms | Single belief |
| **detect_gaps()** | ~200ms | Full scan |
| **learn_from_experience()** | ~10ms | Model update |

---

## ⚙️ Configuration

```python
@dataclass
class ActiveInferenceConfig:
    state_dim: int = 64              # Dimensão do estado interno
    action_embedding_dim: int = 32   # Embedding das ações
    planning_horizon: int = 5        # Passos de lookahead
    num_action_samples: int = 20     # Candidatos por seleção
    temperature: float = 1.0         # Softmax temperature
    risk_weight: float = 1.0         # Peso do risco
    ambiguity_weight: float = 1.0    # Peso da ambiguidade
    novelty_bonus: float = 0.3       # Bônus para novidade
    learning_rate: float = 0.01      # LR do modelo
    belief_decay: float = 0.99       # Decay dos beliefs
```

---

## 🔮 Future Enhancements

- [ ] Hierarchical Active Inference (multi-scale planning)
- [ ] Meta-learning de preferências baseado em feedback
- [ ] Integração com reinforcement learning
- [ ] Ações contínuas (não apenas discretas)

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/learning/free_energy.md
================================================================================

# ⚡ Free Energy Principle

**Module**: `core/learning/free_energy.py`  
**Lines of Code**: 1257  
**Purpose**: Implementação do Princípio de Energia Livre (topo da hierarquia cognitiva)

---

## 🎯 Overview

O módulo Free Energy é o **topo conceitual** de toda a hierarquia cognitiva do Alexandria. Implementa o Princípio de Energia Livre de Karl Friston, que unifica percepção, ação e aprendizado sob uma única métrica: **minimizar energia livre variacional**.

Este módulo integra os outros módulos de learning:
- **Hebbian** → minimiza energia local
- **Meta-Hebbian** → aprende como minimizar
- **Predictive Coding** → minimiza erro de predição
- **Active Inference** → minimiza F esperado via ação

### Conceito Chave: Free Energy

```
F = Complexity - Accuracy
F = D_KL[Q(s) || P(s)] - E_Q[log P(o|s)]

Complexity: Distância entre beliefs Q e prior P
Accuracy: Quão bem o modelo explica observações
```

**Objetivo**: Minimizar F → Beliefs precisos com modelo parcimonioso.

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Components["Free Energy System"]
        VFE[Variational Free Energy]
        EFE[Expected Free Energy]
        UM[Unified Mind]
    end
    
    subgraph Inputs
        O[Observations]
        B[Beliefs Q(s)]
        P[Priors P(s)]
    end
    
    subgraph Outputs
        F[F Score]
        UB[Updated Beliefs]
        UA[Selected Actions]
    end
    
    O --> VFE
    B --> VFE
    P --> VFE
    VFE --> F
    VFE --> UB
    
    B --> EFE
    EFE --> UA
    
    VFE --> UM
    EFE --> UM
    
    style Components fill:#9C27B0,color:#fff
    style Outputs fill:#4CAF50,color:#fff
```

---

## 🔍 Variational Free Energy

### Computação de F

```python
def compute(self, observation: Optional[np.ndarray] = None):
    """
    F = Complexity - Accuracy
    
    Returns:
        F: Energia livre total
        components: Breakdown
    """
    # Complexity: D_KL[Q(s) || P(s)]
    complexity = self._kl_divergence_gaussian(
        self.belief_mean, self.belief_variance,
        self.prior_mean, self.prior_variance
    )
    
    # Accuracy: E_Q[log P(o|s)]
    if observation is not None:
        predicted_obs = self.recognition_model @ self.belief_mean
        accuracy = -0.5 * np.sum((observation - predicted_obs)**2 * self.obs_precision)
    else:
        accuracy = 0.0
    
    F = self.config.complexity_weight * complexity - self.config.accuracy_weight * accuracy
    
    return F, {"complexity": complexity, "accuracy": accuracy}
```

### Atualização de Beliefs

```python
def update_beliefs(self, observation: np.ndarray, learning_rate=None):
    """
    Move beliefs na direção que reduz F.
    
    Δμ = -dF/dμ = precision * (W.T @ (o - W @ μ))
    """
    lr = learning_rate or self.config.belief_learning_rate
    
    # Prediction error
    prediction = self.recognition_model @ self.belief_mean
    error = observation - prediction
    
    # Gradient descent on F
    gradient = self.recognition_model.T @ (self.obs_precision * error)
    self.belief_mean += lr * gradient
    
    # Update precision based on error variance
    self._update_precision(error)
```

---

## 🔍 Expected Free Energy

### Seleção de Ação

```python
class ExpectedFreeEnergy:
    """
    G(π) = D_KL[Q(o|π) || P(o)] + E[H(o|s,π)]
         = Risk + Ambiguity
    """
    
    def compute_for_action(self, action, current_belief):
        # Risk: Distância das preferências
        predicted_outcome = self.model.predict_outcome(action)
        risk = self._kl_divergence(predicted_outcome, self.preferences)
        
        # Ambiguity: Incerteza esperada
        ambiguity = self._expected_entropy(action, current_belief)
        
        G = self.config.risk_weight * risk + self.config.ambiguity_weight * ambiguity
        return G
```

---

## 🔍 Unified Mind

### Integração Completa

```python
class UnifiedMind:
    """
    Sistema que integra todos os componentes:
    - Meta-Hebbian (plasticidade)
    - Predictive Coding (percepção)
    - Active Inference (ação)
    - Free Energy (métrica unificadora)
    """
    
    def cognitive_cycle(self, observation):
        # 1. Percepção via Predictive Coding
        code, pc_stats = self.predictive_coding.process(observation)
        
        # 2. Atualiza beliefs
        self.free_energy.update_beliefs(observation)
        
        # 3. Seleciona ação via Active Inference
        action = self.active_inference.plan()
        
        # 4. Aprende via Meta-Hebbian
        if action.was_successful:
            self.meta_hebbian.evolve_rules(fitness_scores)
        
        return action, self.free_energy.compute()
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    FE[Free Energy] --> MH[Meta-Hebbian]
    FE --> PC[Predictive Coding]
    FE --> AI[Active Inference]
    
    style FE fill:#9C27B0,color:#fff
```

### **Used By**:

```mermaid
graph LR
    UM[Unified Mind] --> FE[Free Energy]
    IL[Integration Layer] --> FE
    
    style FE fill:#9C27B0,color:#fff
```

---

## 📊 Core Classes

### FreeEnergyMode

```python
class FreeEnergyMode(Enum):
    PERCEPTION = auto()   # Só atualiza beliefs
    ACTION = auto()       # Só seleciona ações
    LEARNING = auto()     # Só aprende modelo
    FULL = auto()         # Ciclo completo
```

### FreeEnergyConfig

```python
@dataclass
class FreeEnergyConfig:
    state_dim: int = 64
    observation_dim: int = 384
    complexity_weight: float = 1.0
    accuracy_weight: float = 1.0
    risk_weight: float = 1.0
    ambiguity_weight: float = 1.0
    belief_learning_rate: float = 0.1
    model_learning_rate: float = 0.01
    precision_learning_rate: float = 0.001
    preferred_states: Optional[np.ndarray] = None
    history_length: int = 1000
```

---

## 🎯 Use Cases

### 1. Computar Energia Livre

```python
from core.learning.free_energy import VariationalFreeEnergy, FreeEnergyConfig

vfe = VariationalFreeEnergy(FreeEnergyConfig())

observation = get_embedding("neural plasticity")
F, components = vfe.compute(observation)

print(f"Free Energy: {F:.4f}")
print(f"  Complexity: {components['complexity']:.4f}")
print(f"  Accuracy: {components['accuracy']:.4f}")
```

### 2. Ciclo Cognitivo Completo

```python
from core.learning.free_energy import UnifiedMind

mind = UnifiedMind()

for observation in observation_stream:
    action, F = mind.cognitive_cycle(observation)
    
    if F > threshold:
        print(f"Alta surpresa! F={F:.4f}")
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **compute()** | ~2ms | F calculation |
| **update_beliefs()** | ~5ms | Gradient step |
| **cognitive_cycle()** | ~100ms | Full loop |
| **compute_for_action()** | ~10ms | Per action |

---

## ⚙️ Configuration

```python
# Decomposição 1: Complexity - Accuracy
COMPLEXITY_WEIGHT = 1.0  # Penaliza beliefs complexos
ACCURACY_WEIGHT = 1.0    # Recompensa boas predições

# Decomposição 2: Risk - Ambiguity  
RISK_WEIGHT = 1.0        # Penaliza estados não-preferidos
AMBIGUITY_WEIGHT = 1.0   # Penaliza incerteza

# Learning rates
BELIEF_LR = 0.1          # Update de beliefs
MODEL_LR = 0.01          # Update do modelo generativo
PRECISION_LR = 0.001     # Update de precisão
```

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/learning/integration_layer.md
================================================================================

# 🔌 Integration Layer

**Module**: `core/learning/integration_layer.py`  
**Lines of Code**: 1223  
**Purpose**: Camada de integração que conecta todos os módulos cognitivos

---

## 🎯 Overview

O Integration Layer é o **ponto de conexão** entre todos os módulos de learning. Resolve conflitos de interface entre os diferentes módulos e fornece uma API unificada.

### Conflitos Resolvidos

| Conflito | Problema | Solução |
|----------|----------|---------|
| **A** | Meta-Hebbian assume matriz densa, Mycelial usa Dict esparso | `SparseGraphAdapter` |
| **B** | PC comprime para 32D, VQ-VAE espera 384D | `IsomorphicPredictiveCoding` |
| **C** | Diferentes profiles de recurso | `ResourceLimits` por profile |

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Adapters["Adaptadores"]
        SGA[SparseGraphAdapter]
        IPC[IsomorphicPredictiveCoding]
    end
    
    subgraph Modules["Módulos"]
        MH[Meta-Hebbian]
        PC[Predictive Coding]
        AI[Active Inference]
    end
    
    subgraph Resources["Resource Management"]
        RL[ResourceLimits]
        SP[SystemProfile]
    end
    
    SGA --> MH
    IPC --> PC
    
    style Adapters fill:#2196F3,color:#fff
```

---

## 🔍 SparseGraphAdapter

```python
class SparseGraphAdapter:
    """Permite Meta-Hebbian operar em grafos esparsos"""
    
    def apply_to_sparse_graph(
        self,
        graph: Dict[int, Dict[int, float]],
        activated_nodes: Set[int],
        activations: Optional[Dict[int, float]] = None
    ):
        # Extrai submatriz local
        local_nodes = self._get_local_neighborhood(activated_nodes)
        local_matrix = self._to_dense(graph, local_nodes)
        
        # Aplica Meta-Hebbian
        updated = self.meta_hebbian.compute_weight_update(
            local_matrix,
            pre_activations,
            post_activations
        )
        
        # Converte de volta para esparso
        self._update_sparse(graph, local_nodes, updated)
```

---

## 🔍 IsomorphicPredictiveCoding

```python
class IsomorphicPredictiveCoding:
    """PC que preserva dimensionalidade 384D"""
    
    def process(self, embedding: np.ndarray, learn: bool = True):
        # Processa através do PC interno
        code = self.pc_network.encode(embedding)
        
        # Reconstrói para 384D
        output = self._reconstruct(code)
        
        if learn:
            self.pc_network.learn(embedding)
        
        return output, stats
```

---

## 🔍 System Profiles

```python
class SystemProfile(Enum):
    LITE = auto()        # 2GB RAM, 10k nodes
    BALANCED = auto()    # 8GB RAM, 100k nodes
    PERFORMANCE = auto() # 32GB RAM, 1M nodes

@dataclass
class ResourceLimits:
    max_memory_mb: int
    max_graph_nodes: int
    pruning_threshold: float
    
    @classmethod
    def from_profile(cls, profile: SystemProfile):
        configs = {
            SystemProfile.LITE: (2000, 10000, 0.3),
            SystemProfile.BALANCED: (8000, 100000, 0.1),
            SystemProfile.PERFORMANCE: (32000, 1000000, 0.01)
        }
        return cls(*configs[profile])
```

---

## 📊 IntegrationConfig

```python
@dataclass
class IntegrationConfig:
    profile: SystemProfile = SystemProfile.BALANCED
    embedding_dim: int = 384
    pc_hidden_dims: List[int] = field(default_factory=lambda: [384, 384])
    pc_code_dim: int = 384
    pc_num_iterations: int = 5
    meta_num_heads: int = 4
    meta_evolution_interval: int = 100
    ai_state_dim: int = 64
    ai_planning_horizon: int = 5
    state_dir: str = "data/integration/"
```

---

## 🎯 Use Cases

### 1. Sistema Integrado Completo

```python
from core.learning.integration_layer import IntegrationLayer, IntegrationConfig

config = IntegrationConfig(profile=SystemProfile.BALANCED)
layer = IntegrationLayer(config)

# Processa embedding através de todo o pipeline
result = layer.process(embedding)
```

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/learning/meta_hebbian.md
================================================================================

# 🧬 Meta-Hebbian Plasticity

**Module**: `core/learning/meta_hebbian.py`  
**Lines of Code**: 784  
**Purpose**: Meta-aprendizado via regras de plasticidade Hebbian evoluídas

---

## 🎯 Overview

O Meta-Hebbian implementa **meta-aprendizado**: em vez de otimizar pesos diretamente, otimiza as **regras** que governam como os pesos mudam. Baseado no paper "Meta-Learning through Hebbian Plasticity in Random Networks" (Najarro & Risi).

Isso permite que o sistema **aprenda a aprender** - as regras de plasticidade evoluem para serem mais eficientes para o domínio específico.

### Conceito Chave: Regra ABCD

```
Δw_ij = η × (A×o_i×o_j + B×o_i + C×o_j + D)

A: Termo Hebbian clássico (correlação)
B: Termo pré-sináptico
C: Termo pós-sináptico  
D: Decay/bias
η: Learning rate
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Rules["Plasticity Rules"]
        R1[Rule Head 1: ABCD]
        R2[Rule Head 2: ABCD]
        R3[Rule Head 3: ABCD]
        R4[Rule Head 4: ABCD]
    end
    
    subgraph Evolution["Meta-Learning"]
        ES[Evolution Strategies]
        GD[Gradient Descent]
    end
    
    subgraph Application["Runtime"]
        W[Weight Matrix]
        Pre[Pre-synaptic]
        Post[Post-synaptic]
    end
    
    Rules --> Application
    Application --> Fitness
    Fitness --> Evolution
    Evolution --> Rules
    
    style Rules fill:#4CAF50,color:#fff
    style Evolution fill:#E91E63,color:#fff
```

---

## 🔍 Regras de Plasticidade

```python
@dataclass
class PlasticityRule:
    A: float = 1.0   # Hebbian term
    B: float = 0.0   # Pre-synaptic term
    C: float = 0.0   # Post-synaptic term
    D: float = 0.0   # Decay/constant
    eta: float = 0.01  # Learning rate
    
    def compute_delta(self, pre: float, post: float) -> float:
        return self.eta * (
            self.A * pre * post +
            self.B * pre +
            self.C * post +
            self.D
        )
    
    def compute_delta_matrix(self, pre: np.ndarray, post: np.ndarray):
        """Vectorized para todas as sinapses"""
        outer = np.outer(post, pre)  # [post_size x pre_size]
        return self.eta * (
            self.A * outer +
            self.B * pre[np.newaxis, :] +
            self.C * post[:, np.newaxis] +
            self.D
        )
```

---

## 🔍 Evolução das Regras

### Evolution Strategies

```python
def evolve_rules(self, fitness_scores: List[float]):
    """
    Evolui regras baseado em fitness.
    M(H, T_old) → T_new
    """
    current_fitness = np.mean(fitness_scores[-10:])
    
    if self.config.evolution_method == EvolutionMethod.EVOLUTION_STRATEGIES:
        return self._evolve_es(current_fitness)
    else:
        return self._evolve_gradient(fitness_scores)

def _evolve_es(self, current_fitness: float):
    """
    Evolution Strategies (Najarro & Risi style)
    """
    # Gera população de regras mutadas
    population = []
    for _ in range(self.config.population_size):
        mutated = self._mutate_rules(self.rules, self.config.sigma)
        population.append(mutated)
    
    # Avalia cada candidato
    fitness = [self._evaluate_rules(rules) for rules in population]
    
    # Seleciona elite
    elite_idx = np.argsort(fitness)[-int(self.config.elite_ratio * len(population)):]
    elite = [population[i] for i in elite_idx]
    
    # Nova regra = média da elite
    self.rules = self._average_rules(elite)
```

---

## 🔗 Inter-Module Communication

### **Depends On**:
Nenhuma dependência interna.

### **Used By**:

```mermaid
graph LR
    PC[Predictive Coding] --> MH[Meta-Hebbian]
    AI[Active Inference] --> MH
    IL[Integration Layer] --> MH
    MR[Mycelial Reasoning] --> MH
    
    style MH fill:#4CAF50,color:#fff
```

---

## 📊 Core Classes

### MetaHebbianConfig

```python
@dataclass
class MetaHebbianConfig:
    num_codes: int = 1024         # Tamanho da rede
    num_heads: int = 4            # Regras paralelas
    rules_per_head: bool = True   # Regra diferente por head
    evolution_method: EvolutionMethod = EvolutionMethod.EVOLUTION_STRATEGIES
    population_size: int = 50
    sigma: float = 0.1            # Mutation strength
    elite_ratio: float = 0.2
    param_bounds: Tuple = (-2.0, 2.0)
    eta_bounds: Tuple = (0.0001, 0.1)
```

---

## 🎯 Use Cases

### 1. Aplicar a Rede Micelial

```python
from core.learning.meta_hebbian import MetaHebbianPlasticity

mh = MetaHebbianPlasticity()

# Aplica regras às conexões
new_connections = mh.apply_to_mycelial(
    connections=mycelial_network.connections,
    activated_codes=active_codes,
    activation_strengths=strengths
)
```

### 2. Evoluir Regras

```python
# Após ciclo de aprendizado
fitness_scores = [0.7, 0.75, 0.8, 0.82]  # Histórico
stats = mh.evolve_rules(fitness_scores)

print(f"Nova regra: A={mh.rules[0].A:.3f}, B={mh.rules[0].B:.3f}")
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **compute_weight_update()** | ~2ms | Single matrix |
| **apply_to_mycelial()** | ~50ms | Sparse graph |
| **evolve_rules()** | ~500ms | ES population=50 |

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/learning/predictive_coding.md
================================================================================

# 🔮 Predictive Coding

**Module**: `core/learning/predictive_coding.py`  
**Lines of Code**: 981  
**Purpose**: Rede hierárquica de codificação preditiva para inferência

---

## 🎯 Overview

O Predictive Coding implementa uma **rede neural hierárquica** onde cada camada tenta prever a atividade da camada abaixo. Os **erros de predição** são propagados para cima, enquanto **predições** fluem para baixo.

Este é o mecanismo de **percepção** do Alexandria - não é feedforward tradicional, mas um processo iterativo que minimiza energia livre através de settling dinâmico.

### Conceito Chave

```
Erro = Precisão × (Input - Predição)
     ε = Π × (o - μ)

Input bottom-up: o que realmente observamos
Predição top-down: o que esperávamos ver
Precisão: confiança no erro
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Hierarchy["Predictive Coding Network"]
        direction TB
        L0[Input 384D]
        L1[Layer 1: 256D]
        L2[Layer 2: 128D]
        L3[Layer 3: 64D]
        L4[Code: 32D]
    end
    
    L0 -->|"error ε"| L1
    L1 -->|"error ε"| L2
    L2 -->|"error ε"| L3
    L3 -->|"error ε"| L4
    
    L4 -->|"prediction μ"| L3
    L3 -->|"prediction μ"| L2
    L2 -->|"prediction μ"| L1
    L1 -->|"prediction μ"| L0
    
    style L0 fill:#4CAF50,color:#fff
    style L4 fill:#2196F3,color:#fff
```

---

## 🔍 Camada Preditiva

Cada `PredictiveLayer` implementa:

```python
class PredictiveLayer:
    def predict(self, top_down_input=None):
        """
        Gera predição para camada abaixo.
        prediction = W_pred @ representation + bias + noise
        """
        if top_down_input is not None:
            self.representation = top_down_input
        
        prediction = self.W_pred @ self.representation + self.bias
        prediction += np.random.normal(0, self.config.prediction_noise, prediction.shape)
        return prediction
    
    def compute_error(self, bottom_up_input):
        """
        Computa erro de predição precision-weighted.
        error = precision * (input - prediction)
        """
        prediction = self.predict()
        raw_error = bottom_up_input - prediction
        error = self.precision * raw_error
        return error
    
    def update_representation(self, bottom_up_error, top_down_prediction=None):
        """
        Atualiza representação para minimizar erro.
        
        Δμ = lr * (W_err @ ε_below - ε_self)
        """
        for _ in range(self.config.num_iterations):
            # Erro vindo de baixo
            gradient = self.W_err @ bottom_up_error
            
            # Erro próprio (constraint top-down)
            if top_down_prediction is not None:
                self_error = self.representation - top_down_prediction
                gradient -= self_error
            
            self.representation += self.config.learning_rate * gradient
```

---

## 🔍 Rede Completa

### Processo de Inferência

```python
class PredictiveCodingNetwork:
    def encode(self, observation, num_iterations=10):
        """
        Codifica observação através de settling iterativo.
        
        NÃO é um forward pass - é um processo de energia mínima.
        """
        # Inicializa com forward pass
        x = observation
        for layer in self.layers:
            layer.representation = layer.W_init @ x
            x = layer.representation
        
        # Settling: itera até convergir
        for _ in range(num_iterations):
            # Top-down: predições
            for i in range(len(self.layers) - 1, 0, -1):
                prediction = self.layers[i].predict()
                self.layers[i-1].top_down_prediction = prediction
            
            # Bottom-up: erros
            errors = [None] * len(self.layers)
            errors[0] = self.layers[0].compute_error(observation)
            for i in range(1, len(self.layers)):
                errors[i] = self.layers[i].compute_error(self.layers[i-1].representation)
            
            # Update representations
            for i, layer in enumerate(self.layers):
                layer.update_representation(
                    errors[i],
                    layer.top_down_prediction if hasattr(layer, 'top_down_prediction') else None
                )
        
        return self.layers[-1].representation
```

### Aprendizado

```python
def learn(self, observation):
    """
    Atualiza pesos para melhorar predições futuras.
    
    ΔW_pred = lr * ε @ μ.T (Hebbian no erro!)
    """
    code = self.encode(observation)
    
    for i, layer in enumerate(self.layers):
        if i == 0:
            target = observation
        else:
            target = self.layers[i-1].representation
        
        error = target - layer.predict()
        
        # Hebbian update on prediction weights
        layer.W_pred += self.config.learning_rate * np.outer(error, layer.representation)
        
        # Update precision
        layer._update_precision(error)
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    PC[Predictive Coding] --> MH[Meta-Hebbian]
    
    style PC fill:#FF9800,color:#fff
```

**Meta-Hebbian** (`meta_hebbian.py`)
- **Purpose**: Regras de plasticidade adaptativas
- **Call**: `meta_hebbian.apply_to_layer()`
- **When**: Opcional, para plasticidade avançada

### **Used By**:

```mermaid
graph LR
    FE[Free Energy] --> PC[Predictive Coding]
    IL[Integration Layer] --> PC
    
    style PC fill:#FF9800,color:#fff
```

---

## 📊 Core Classes

### PrecisionMode

```python
class PrecisionMode(str, Enum):
    FIXED = "fixed"        # Precisão constante
    LEARNED = "learned"    # Precisão aprendida por camada
    ADAPTIVE = "adaptive"  # Precisão ajustada online por erro
```

### PredictiveCodingConfig

```python
@dataclass
class PredictiveCodingConfig:
    input_dim: int = 384
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    code_dim: int = 32
    num_iterations: int = 10         # Iterações de settling
    learning_rate: float = 0.01
    inference_rate: float = 0.1      # Taxa para atualizar representations
    precision_mode: PrecisionMode = PrecisionMode.ADAPTIVE
    base_precision: float = 1.0
    precision_lr: float = 0.001
    use_meta_hebbian: bool = True
    prediction_noise: float = 0.01
```

---

## 🎯 Use Cases

### 1. Encode Embedding

```python
from core.learning.predictive_coding import PredictiveCodingNetwork

pc = PredictiveCodingNetwork()

embedding = get_embedding("neuromodulation in learning")
code = pc.encode(embedding, num_iterations=10)

print(f"Input: {embedding.shape} → Code: {code.shape}")
# Input: (384,) → Code: (32,)
```

### 2. Learn from Data

```python
for embedding in corpus_embeddings:
    # Encode and learn
    code = pc.encode(embedding)
    pc.learn(embedding)

# Save learned weights
pc.save_state("models/pc_learned.pkl")
```

### 3. Get Surprise (Anomaly Detection)

```python
# Quanto o input difere das predições
surprise = pc.get_surprise(new_embedding)

if surprise > threshold:
    print(f"Anomalia detectada! Surprise: {surprise:.4f}")
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **encode()** | ~15ms | 10 iterations |
| **learn()** | ~5ms | Single sample |
| **get_surprise()** | ~3ms | No learning |
| **save_state()** | ~50ms | Full network |

---

## ⚙️ Configuration

```python
# Arquitetura
INPUT_DIM = 384                  # Sentence-Transformer output
HIDDEN_DIMS = [256, 128, 64]     # Compressão gradual
CODE_DIM = 32                    # Representação final

# Inferência
NUM_ITERATIONS = 10              # Passos de settling
INFERENCE_RATE = 0.1             # Step size para representation update

# Precisão
PRECISION_MODE = "adaptive"      # Aprende online
BASE_PRECISION = 1.0
PRECISION_LR = 0.001
```

---

## 🔮 Future Enhancements

- [ ] Attention-weighted precision
- [ ] Temporal predictive coding (sequências)
- [ ] Sparse coding constraints
- [ ] GPU acceleration (PyTorch backend)

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/learning/profiles.md
================================================================================

# 🎭 Reasoning Profiles

**Module**: `core/learning/profiles.py`  
**Lines of Code**: 102  
**Purpose**: Personalidades cognitivas para Multi-Agent NEMESIS

---

## 🎯 Overview

Define **perfis de raciocínio** distintos para a arquitetura multi-agente. Cada perfil representa uma estratégia diferente para explorar e atualizar a memória Mycelial compartilhada.

---

## 🤖 Perfis Disponíveis

### 🔭 The Scout (Explorador)

```python
ReasoningProfile(
    name="The Scout",
    description="High-speed explorer of the unknown.",
    risk_weight=0.1,         # Baixo medo de errar
    ambiguity_weight=0.5,    # Interesse moderado em clareza
    novelty_bonus=2.0,       # Alta busca por novidade
    planning_horizon=2,      # Tático, curto prazo
    temperature=2.0,         # Alta aleatoriedade
    learning_rate_mod=1.5,   # Aprende rápido
    max_steps_per_cycle=20
)
```

**Papel**: Gerar hipóteses rapidamente, encontrar papers/conceitos novos.

---

### ⚖️ The Judge (Verificador)

```python
ReasoningProfile(
    name="The Judge",
    description="Critical verifier of truth/consistency.",
    risk_weight=5.0,         # Odeia estar errado
    ambiguity_weight=2.0,    # Precisa resolver incerteza
    novelty_bonus=-0.5,      # Penaliza novidades
    planning_horizon=8,      # Pensamento estratégico profundo
    temperature=0.1,         # Determinístico
    learning_rate_mod=0.2,   # Difícil mudar de opinião
    max_steps_per_cycle=5
)
```

**Papel**: Verificar conexões existentes, remover as fracas.

---

### 🕸️ The Weaver (Conector)

```python
ReasoningProfile(
    name="The Weaver",
    description="Architect of long-range connections.",
    risk_weight=1.0,
    ambiguity_weight=1.0,
    novelty_bonus=0.5,       # Balanceado
    planning_horizon=5,
    temperature=0.8,
    learning_rate_mod=1.0,
    max_steps_per_cycle=10
)
```

**Papel**: Encontrar gaps estruturais, conectar clusters distantes.

---

## 📊 Comparação

| Parâmetro | Scout | Judge | Weaver |
|-----------|-------|-------|--------|
| Risk Weight | 0.1 | 5.0 | 1.0 |
| Novelty Bonus | 2.0 | -0.5 | 0.5 |
| Planning Horizon | 2 | 8 | 5 |
| Temperature | 2.0 | 0.1 | 0.8 |
| Steps/Cycle | 20 | 5 | 10 |

---

## 🎯 Use Cases

```python
from core.learning.profiles import get_scout_profile, get_judge_profile

# Usa Scout para exploração
scout = get_scout_profile()
agent.set_profile(scout)

# Muda para Judge para verificação
judge = get_judge_profile()
agent.set_profile(judge)
```

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/loop/README.md
================================================================================

# 🔄 Self-Feeding Loop

**Module**: `core/loop/`  
**Lines of Code**: ~1,240  
**Purpose**: Ciclo auto-alimentado que conecta raciocínio, ação e aprendizado

---

## 🎯 Overview

O Self-Feeding Loop implementa um ciclo cognitivo fechado:

ORCHESTRATION:
```
semantic_memory → vqvae → mycelial → abduction → nemesis (active_inference)
       ↑                                            ↓
       └──── neural_learner ← feedback ← action ←───┘
```

### Componentes

| Componente | Função |
|------------|--------|
| `HypothesisExecutor` | Transforma hipóteses em ações |
| `ActionFeedbackCollector` | Coleta feedback das ações |
| `NemesisIntegration` | **Active Inference**: Seleciona ações por EFE e fecha loop de Predictive Coding |
| `IncrementalLearner` | Acumula e dispara treinamento |
| `SelfFeedingLoop` | Orquestrador principal |
| `LoopMetrics` | Tracking de performance |

---

## 🚀 Quick Start

```python
from core.loop import SelfFeedingLoop, LoopConfig

# Configurar
config = LoopConfig(
    max_cycles=50,
    stop_on_convergence=True
)

# Criar loop
loop = SelfFeedingLoop(
    abduction_engine=my_abduction,  # opcional
    config=config
)

# Executar
results = loop.run_continuous()
print(f"Ciclos: {results['cycles_run']}")
print(f"Convergiu: {results['converged']}")
```

---

## 📊 Métricas

```python
# Ver métricas
summary = loop.metrics.get_summary()
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Convergence: {summary['convergence_score']:.2f}")

# Salvar métricas
loop.metrics.save_to_file("data/loop_metrics.json")
```

---

## 🔧 Configuração

```python
@dataclass
class LoopConfig:
    max_hypotheses_per_cycle: int = 5
    max_cycles: int = 100
    stop_on_convergence: bool = True
    convergence_threshold: float = 0.01
    min_confidence_threshold: float = 0.1
```

---

**Last Updated**: 2025-12-07


---


# 📄 FILE: docs/core/loop/feedback_collector.md
================================================================================

# 🔄 Feedback Collector

**Module**: `core/loop/feedback_collector.py`
**Class**: `ActionFeedbackCollector`

---

## Propósito

O `ActionFeedbackCollector` é responsável por fechar o ciclo de aprendizado, transformando o resultado bruto das ações (`ActionResult`) em sinais de treinamento (`TrainingFeedback`) para a camada neural (Neural Learner).

Ele atua como um "crítico" que avalia se uma ação foi bem-sucedida e qual a magnitude de sua recompensa.

---

## Lógica de Recompensa (Reward Shaping)

A função `_calculate_reward` implementa a heurística de recompensa:

1.  **Falha na Ação**: Reward fixo negativo (`-0.5`).
2.  **Sucesso sem Evidência**: Reward neutro (`0.0`).
3.  **Sucesso com Evidência**: Base positiva (`0.5`) + Proporcional à evidência encontrada (até `1.0`).
4.  **Bônus de Conexão**: Adicional de `+0.3` se a ação gerou novas arestas no grafo causal.

### Fórmula
```python
reward = base_reward + (evidence_score * 0.5) + connection_bonus
```
Limitado por `min_reward` e `max_reward`.

---

## Estrutura de Dados

### `TrainingFeedback`

Objeto padronizado enviado para o `IncrementalLearner`:

- `embeddings`: Lista de vetores das evidências encontradas (para treino contrastivo).
- `reward_signal`: Scalar float indicando qualidade da ação (-1.0 a +1.0).
- `should_learn`: Booleano indicando se o feedback é significativo o suficiente para disparar backprop.
- `source_action_type`: Tipo da ação que gerou o feedback.

---

## API Reference

### `collect`

```python
def collect(self, action_result: Dict) -> TrainingFeedback
```

Processa o resultado da ação, calcula reward e extrai embeddings se houver topologia disponível.

### `get_stats`

```python
def get_stats(self) -> Dict
```

Retorna métricas acumuladas:
- Total de feedbacks coletados.
- Taxa de feedbacks positivos.
- Reward médio (janela móvel).

---

## Exemplo de Integração

```python
# No loop principal:
result = executor.execute(hypothesis)
feedback = collector.collect(result.to_dict())

if feedback.should_learn:
    learner.train(feedback)
```


---


# 📄 FILE: docs/core/loop/hypothesis_executor.md
================================================================================

# HypothesisExecutor (`core/loop/hypothesis_executor.py`)

> Transforma hipóteses do AbductionEngine em ações executáveis.

## Visão Geral

O `HypothesisExecutor` é a ponte entre raciocínio abstrato (hipóteses) e ações concretas. Ele converte hipóteses geradas pelo `AbductionEngine` em ações tipadas que podem ser executadas pelo sistema.

## Arquitetura

```
AbductionEngine          HypothesisExecutor         ActionAgent
     │                          │                        │
     │  hypothesis              │                        │
     └─────────────────────────→│                        │
                                │  ExecutableAction      │
                                └───────────────────────→│
                                                         │
                                     ActionResult        │
                                ←────────────────────────┘
```

## Classes

### ExecutionActionType

Enum de tipos de ação baseados em confiança:

| Tipo | Confiança | Descrição |
|------|-----------|-----------|
| `QUERY_SEARCH` | < 0.3 | Precisa mais evidência |
| `EXPLORE_CLUSTER` | 0.3-0.5 | Explora cluster relacionado |
| `BRIDGE_CONCEPTS` | 0.5-0.8 | Conecta conceitos |
| `VALIDATE_EXISTING` | 0.8-1.0 | Valida hipótese existente |
| `DEEPEN_TOPIC` | > 0.8 | Aprofunda tópico |

### ExecutableAction

```python
@dataclass
class ExecutableAction:
    action_type: ExecutionActionType
    target: str
    parameters: Dict[str, Any]
    expected_outcome: str
    source_hypothesis_id: str
    priority: float
```

### ActionResult

```python
@dataclass
class ActionResult:
    action: ExecutableAction
    success: bool
    evidence_found: List[str]
    new_connections: int
    execution_time_ms: float
    error_message: str
```

## Uso

```python
from core.loop.hypothesis_executor import HypothesisExecutor

executor = HypothesisExecutor(
    semantic_memory=memory,
    topology_engine=engine
)

# Converter hipótese em ação
hypothesis = {
    "hypothesis_text": "Meta-learning usa Free Energy",
    "source_cluster": "meta-learning",
    "target_cluster": "free-energy",
    "confidence_score": 0.6
}

action = executor.hypothesis_to_action(hypothesis)
# → ExecutableAction(type=BRIDGE_CONCEPTS, ...)

# Executar
result = executor.execute(hypothesis)
print(f"Sucesso: {result.success}")
print(f"Evidências: {result.evidence_found}")
```

## Regras de Mapeamento

```
confidence < 0.3     → QUERY_SEARCH
0.3 <= conf < 0.5    → EXPLORE_CLUSTER
0.5 <= conf < 0.8    → BRIDGE_CONCEPTS
conf >= 0.8          → VALIDATE_EXISTING ou DEEPEN_TOPIC
```

## Estatísticas

```python
stats = executor.get_stats()
# {
#     "total_executed": 42,
#     "successful": 35,
#     "success_rate": 0.83,
#     "avg_execution_time_ms": 150.2
# }
```


---


# 📄 FILE: docs/core/loop/incremental_learner.md
================================================================================

# IncrementalLearner (`core/loop/incremental_learner.py`)

> Acumula feedback e dispara treinamento em batches.

## Visão Geral

O `IncrementalLearner` gerencia o aprendizado contínuo do VQ-VAE. Acumula feedback em batches e dispara treinamento quando atinge thresholds configurados.

## Fluxo

```
Feedback     IncrementalLearner        V2Learner
    │                │                     │
    └───add_feedback→│                     │
                     │  [acumula]          │
                     │                     │
                     │  batch >= threshold │
                     │                     │
                     └────learn()─────────→│
                                           │ treina modelo
                     ←───────metrics───────┘
```

## Classes

### LearningSession

Registro de uma sessão de aprendizado:

```python
@dataclass
class LearningSession:
    timestamp: str
    batch_size: int
    total_loss: float
    recon_loss: float
    vq_loss: float
    avg_reward: float
```

### IncrementalLearner

```python
learner = IncrementalLearner(
    v2_learner=v2_model,
    batch_threshold=10,      # Treina quando acumula 10 embeddings
    reward_threshold=3.0,    # Ou quando reward acumulado > 3
    auto_save=True           # Salva modelo após cada sessão
)
```

## Uso

```python
from core.loop.incremental_learner import IncrementalLearner

learner = IncrementalLearner(batch_threshold=10)

# Adicionar feedback (retorna True se aprendizado foi triggado)
triggered = learner.add_feedback({
    "embeddings": [emb1, emb2, emb3],
    "reward_signal": 0.8,
    "should_learn": True
})

# Forçar aprendizado
metrics = learner.force_learn()

# Estatísticas
stats = learner.get_stats()
# {
#     "total_learned": 150,
#     "current_batch_size": 5,
#     "accumulated_reward": 2.3,
#     "last_loss": 0.012,
#     "sessions_count": 15,
#     "avg_loss": 0.015
# }

# Histórico
history = learner.get_learning_history()
```

## Triggers de Aprendizado

O aprendizado é disparado quando:

1. `batch_size >= batch_threshold` (default: 10)
2. `accumulated_reward >= reward_threshold` (default: 3.0)
3. `force_learn()` é chamado manualmente

## Integração

Usado pelo `SelfFeedingLoop` no ciclo:

```
observe → generate → execute → feedback → learn (IncrementalLearner)
                                              ↓
                                         V2Learner.learn()
```


---


# 📄 FILE: docs/core/loop/loop_metrics.md
================================================================================

# LoopMetrics (`core/loop/loop_metrics.py`)

> Tracking de performance e detecção de convergência.

## Visão Geral

O `LoopMetrics` coleta, agrega e analisa métricas de cada ciclo do Self-Feeding Loop. Permite monitorar performance e detectar quando o sistema convergiu.

## Classes

### CycleMetrics

Métricas de um único ciclo:

```python
@dataclass
class CycleMetrics:
    cycle_id: int
    timestamp: str
    
    # Detecção
    gaps_detected: int
    hypotheses_generated: int
    
    # Execução
    actions_executed: int
    actions_successful: int
    
    # Feedback
    total_evidence: int
    new_connections: int
    avg_reward: float
    
    # Aprendizado
    learning_triggered: bool
    loss: float
    
    # Timing
    cycle_time_ms: float
```

### LoopMetrics

```python
metrics = LoopMetrics(
    convergence_window=10,      # Janela para calcular convergência
    convergence_threshold=0.01  # Variância máxima para convergir
)
```

## Uso

```python
from core.loop.loop_metrics import LoopMetrics, CycleMetrics

metrics = LoopMetrics()

# Iniciar ciclo
cycle = metrics.start_cycle()

# ... executa ciclo ...
cycle.gaps_detected = 5
cycle.hypotheses_generated = 3
cycle.actions_executed = 3
cycle.actions_successful = 2
cycle.avg_reward = 0.75

# Registrar
metrics.record_cycle(cycle)

# Verificar convergência
if metrics.is_converged():
    print("Sistema convergiu!")

# Resumo
summary = metrics.get_summary()
# {
#     "total_cycles": 42,
#     "total_gaps": 210,
#     "success_rate": 0.78,
#     "convergence_score": 0.92,
#     "is_converged": True
# }

# Tendência
trend = metrics.get_trend("avg_reward", window=10)
# > 0 = melhorando, < 0 = piorando
```

## Convergência

O sistema é considerado **convergido** quando:
- Loss tem baixa variância nas últimas N iterações
- `variância(loss) < threshold`

```python
score = metrics.get_convergence_score()  # 0.0 a 1.0
```

## Exportação

```python
# JSON
json_str = metrics.to_json()

# Arquivo
metrics.save_to_file("data/metrics/run_001.json")
```

## Métricas Agregadas

| Métrica | Descrição |
|---------|-----------|
| `total_cycles` | Total de ciclos executados |
| `total_gaps` | Gaps detectados |
| `total_hypotheses` | Hipóteses geradas |
| `total_actions` | Ações executadas |
| `success_rate` | Taxa de sucesso |
| `cumulative_reward` | Reward acumulado |
| `convergence_score` | 0-1, estabilidade do loss |


---


# 📄 FILE: docs/core/loop/nemesis_integration.md
================================================================================

# 🧠 Nemesis Integration

**Module**: `core/loop/nemesis_integration.py`
**Type**: Integration Logic / Cognitive Control

---

## Overview

A classe `NemesisIntegration` atua como o "cérebro executivo" do Loop Auto-Alimentado. Ela encapsula a complexidade teórica dos módulos de Active Inference, Predictive Coding e Free Energy Principle, expondo uma interface pragmática para o `SelfFeedingLoop`.

A principal responsabilidade é **tomar decisões** (selecionar ações) que minimizem a Energia Livre Esperada (EFE) e **aprender** com as consequências dessas ações (atualizar o modelo generativo).

## Interface Principal

### `select_action(gaps, hypotheses) -> Hypothesis`

Seleciona a melhor hipótese a ser agida com base no princípio da Inferência Ativa.

- **Input**: Lista de Gaps de conhecimento e Hipóteses geradas pelo Abduction Engine.
- **Lógica**:
    1. Calcula **Valor Epistêmico** (Ambiguity reduction) para cada hipótese.
    2. Calcula **Valor Pragmático** (Risk/Preference) baseado nos objetivos do sistema.
    3. Combina em **Expected Free Energy (EFE)**.
    4. Seleciona a hipótese com menor EFE.
- **Output**: A hipótese "vencedora" enriquecida com metadados do Nemesis (`nemesis_efe`, etc).

### `update_after_action(action, observation, reward)`

Fecha o ciclo de feedback, permitindo que o sistema aprenda.

- **Input**:
    - `action`: A ação executada.
    - `observation`: O resultado perceptivo (embedding) da ação.
    - `reward`: Sinal de recompensa escalar (sucesso/falha).
- **Processo**:
    1. **Predictive Coding**: Compara a observação prevista com a real → Gera `Prediction Error`.
    2. **Free Energy**: Calcula a Energia Livre Variacional (VFE) do estado atual.
    3. **Meta-Hebbian**: Atualiza pesos sinápticos e taxas de aprendizado baseado na correlação entre erro e recompensa.

## Integração no Loop

O `NemesisIntegration` não roda em isolamento. Ele é injetado no `SelfFeedingLoop` e consultado em dois momentos críticos:

1. **Antes da Execução**: Para filtrar e priorizar hipóteses (`select_best_hypothesis`).
2. **Após a Execução**: Via callback `on_action_complete` para assimilar o resultado.

---

## Exemplo de Uso

```python
from core.loop.nemesis_integration import NemesisIntegration, NemesisConfig

# 1. Configurar
config = NemesisConfig(
    active_inference_enabled=True,
    predictive_coding_enabled=True,
    free_energy_tracking=True
)

# 2. Instanciar
nemesis = NemesisIntegration(config)

# 3. Usar no Loop
# (Geralmente feito automaticamente pelo SelfFeedingLoop se configurado)
best_hyp = nemesis.select_action(gaps, hypotheses)
print(f"Acão selecionada: {best_hyp['hypothesis_text']} (EFE: {best_hyp['nemesis_efe']})")

# 4. Atualizar após ação
nemesis.update_after_action(action_obj, observation_vec, reward_val)
```

---

## Métricas Monitoradas

O módulo exporta métricas vitais para o `LoopMetrics`:

- `free_energy`: Medida de "surpresa" ou desajuste do modelo.
- `prediction_error`: Erro bruto da predição sensorial.
- `model_complexity`: Custo de complexidade das crenças internas.
- `accuracy`: Precisão das predições passadas.
- `efe_history`: Histórico de EFE das ações selecionadas.


---


# 📄 FILE: docs/core/loop/self_feeding_loop.md
================================================================================

# 🔄 Self-Feeding Loop

**Module**: `core/loop/self_feeding_loop.py`
**Class**: `SelfFeedingLoop`

---

## Observação Geral

O `SelfFeedingLoop` é o orquestrador central do sistema cognitivo da Alexandria. Ele implementa um ciclo de controle fechado (closed-loop control) que integra percepção, raciocínio, ação e aprendizado contínuo.

## Arquitetura do Ciclo

O loop executa indefinidamente (ou até critério de parada) seguindo este fluxo:

1.  **Detect Gaps** (`abduction_engine`): Identifica lacunas no grafo de conhecimento.
2.  **Generate Hypotheses** (`abduction_engine`): Produz explicações candidatas para preencher as lacunas.
3.  **Filter**: Seleciona hipóteses com base em confiança (`min_confidence_threshold`).
4.  **Execute Actions** (`hypothesis_executor`): Testa as hipóteses no mundo real ou via simulação.
5.  **Collect Feedback** (`action_feedback_collector`): Avalia o resultado (sucesso, recompensa).
6.  **Incremental Learning** (`incremental_learner`): Atualiza os modelos neurais (VQ-VAE/LLM) com a nova experiência.
7.  **Nemesis Update** (`nemesis_integration`): Se configurado, atualiza o estado de Active Inference (Free Energy).

---

## Configuração (`LoopConfig`)

A dataclass `LoopConfig` controla o comportamento do loop:

| Parâmetro | Tipo | Default | Descrição |
| :--- | :--- | :--- | :--- |
| `max_cycles` | `int` | `100` | Limite máximo de iterações. |
| `max_hypotheses_per_cycle` | `int` | `5` | Máximo de ações por ciclo. |
| `stop_on_convergence` | `bool` | `True` | Se deve parar quando o erro estabilizar. |
| `convergence_threshold` | `float` | `0.01` | Delta mínimo para considerar convergência. |
| `min_confidence_threshold` | `float` | `0.1` | Corte para aceitar hipóteses. |

---

## API Reference

### `__init__`

```python
def __init__(
    self,
    abduction_engine=None,
    hypothesis_executor: Optional[HypothesisExecutor] = None,
    feedback_collector: Optional[ActionFeedbackCollector] = None,
    incremental_learner: Optional[IncrementalLearner] = None,
    config: Optional[LoopConfig] = None,
    on_cycle_complete: Optional[Callable] = None,
    on_action_complete: Optional[Callable] = None
)
```

Inicializa o loop com injeção de dependência. Callbacks permitem observabilidade externa (ex: Nemesis).

### `run_continuous`

```python
def run_continuous(self, max_cycles=None, stop_on_convergence=None) -> Dict
```

Inicia a execução síncrona do loop. Bloqueia até terminar. Retorna um sumário da execução.

### `run_cycle`

```python
def run_cycle(self) -> CycleMetrics
```

Executa uma única iteração (passo) do loop. Útil para execução controlada passo-a-passo.

---

## Exemplo de Uso

```python
from core.loop import SelfFeedingLoop, LoopConfig
from core.reasoning import AbductionEngine

# 1. Configurar
config = LoopConfig(max_cycles=50)

# 2. Instanciar componentes
engine = AbductionEngine()
loop = SelfFeedingLoop(abduction_engine=engine, config=config)

# 3. Executar
summary = loop.run_continuous()

print(f"Terminou após {summary['cycles_run']} ciclos.")
```

---

## Integração com Nemesis

O loop suporta integração com o módulo `Nemesis` através do callback `on_action_complete`. Isso permite que o Nemesis observe as ações e atualize seus modelos de Energia Livre sem acoplar diretamente a lógica no `SelfFeedingLoop`.

```python
def nemesis_callback(hypothesis, result, feedback):
    nemesis.update_after_action(hypothesis, result, feedback)

loop = SelfFeedingLoop(..., on_action_complete=nemesis_callback)
```


---


# 📄 FILE: docs/core/main_api.md
================================================================================

# 🚀 Alexandria API Documentation

**Module**: `main.py`
**Type**: FastAPI Application
**Version**: 12.0

---

## Overview
The entry point for the Alexandria Cognitive System. It exposes a REST API to interact with the Pre-Structural Field and trigger conceptual activation.

## Endpoints

### `GET /`
Returns system status.
- **Response**:
  ```json
  {
    "system": "Alexandria",
    "status": "online",
    "field_initialized": boolean
  }
  ```

### `GET /health`
Health check endpoint.
- **Response**: `{"status": "healthy"}` or `{"status": "degraded", "reason": "..."}`

### `POST /trigger`
Triggers a concept in the semantic manifold.

- **Request Body** (`TriggerRequest`):
  ```json
  {
    "embedding": [float, ...],
    "intensity": float (default: 1.0)
  }
  ```

- **Response**:
  ```json
  {
    "status": "triggered",
    "attractors": int,
    "free_energy": float
  }
  ```

## Configuration
- Initializes `PreStructuralField` on startup.
- Uses `config.settings` for manifold dimensions.

## Error Handling
- Returns `503 Service Unavailable` if the field is not initialized.
- Returns `500 Internal Server Error` for trigger failures.


---


# 📄 FILE: docs/core/memory/semantic_memory.md
================================================================================

# 📚 Semantic Memory System

**Module**: `core/memory/semantic_memory.py`  
**Lines of Code**: 488  
**Purpose**: Multi-modal document indexing and retrieval

---

## 🎯 Overview

The Semantic Memory System is Alexandria's **core storage layer**, responsible for ingesting, processing, and indexing documents (text & images) into a searchable vector database.

### Key Responsibilities

1. **Multi-modal ingestion**: PDFs, text files, images
2. **Intelligent chunking**: Paragraph-aware text segmentation  
3. **Vector embedding**: Convert chunks to 384D semantic vectors
4. **Storage**: LanceDB vector database with metadata
5. **Retrieval**: Semantic similarity search

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input
        A[Document File]
    end
    
    subgraph Router
        B{File Type Detection}
    end
    
    subgraph Text Pipeline
        C[Extract Text]
        D[Chunk Text<br/>~1000 chars]
        E[SentenceTransformer<br/>all-MiniLM-L6-v2]
        F[384D Embeddings]
    end
    
    subgraph Image Pipeline
        G[Load Image]
        H[V11 Vision Encoder]
        I[384D Embedding]
    end
    
    subgraph Storage
        J[LanceDB Insert]
        K[(Vector Database)]
        L[Metadata Store]
    end
    
    A --> B
    B -->|Text/PDF| C
    B -->|Image| G
    C --> D
    D --> E
    E --> F
    G --> H
    H --> I
    F --> J
    I --> J
    J --> K
    J --> L
    
    style Input fill:#e3f2fd
    style Router fill:#fff3e0
    style Text Pipeline fill:#e8f5e9
    style Image Pipeline fill:#f3e5f5
    style Storage fill:#fce4ec
```

---

## 📋 Core Classes

### 1. **SemanticFileSystem**

Main class for multi-modal indexing and retrieval.

```python
class SemanticFileSystem:
    def __init__(self, topology_engine, engine_encoder=None):
        """
        Args:
            topology_engine: TopologyEngine instance (for embeddings)
            engine_encoder: Optional SentenceTransformer model
        """
        self.topology = topology_engine
        self.engine_encoder = engine_encoder
        self.vision_loader = VisionLoader()  # For images
        self.image_processor = ImageProcessor(self.vision_loader)
```

**Key Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `index_file(file_path, doc_type)` | Index document (auto-detects type) | `int` (chunks indexed) |
| `retrieve(query, modality_filter, limit)` | Semantic search | `List[Dict]` (results) |
| `get_stats()` | System statistics | `Dict` |

---

## 🔄 Data Flow

### Indexing Flow

```mermaid
sequenceDiagram
    participant User
    participant SFS as SemanticFileSystem
    participant Router as File Router
    participant Text as Text Processor
    participant Vision as V11 Encoder
    participant DB as LanceDB
    
    User->>SFS: index_file("paper.pdf")
    SFS->>Router: Detect file type
    Router->>Router: Check extension
    
    alt PDF/Text
        Router->>Text: Process as text
        Text->>Text: Extract text
        Text->>Text: Chunk (~1000 chars)
        loop Each chunk
            Text->>Text: Generate embedding (384D)
            Text->>DB: Insert vector + metadata
        end
    else Image
        Router->>Vision: Process as image
        Vision->>Vision: Load & preprocess
        Vision->>Vision: V11 encode (384D)
        Vision->>DB: Insert vector + metadata
    end
    
    DB-->>SFS: Chunks indexed
    SFS-->>User: Return count
```

### Retrieval Flow

```mermaid
sequenceDiagram
    participant User
    participant SFS as SemanticFileSystem
    participant Topology as TopologyEngine
    participant DB as LanceDB
    
    User->>SFS: retrieve("quantum computing")
    SFS->>Topology: encode(query)
    Topology->>Topology: Generate 384D embedding
    Topology-->>SFS: Query vector
    
    SFS->>DB: similarity_search(vector, limit=10)
    DB->>DB: IVF-PQ index search
    DB-->>SFS: Top-K results
    
    SFS->>SFS: Format results
    SFS-->>User: [{content, score, metadata}]
```

---

## 🧩 Components

### **VisionLoader**

Loads and manages V11 Vision Encoder for image processing.

```python
class VisionLoader:
    def load_model(self):
        """
        Lazy-loads V11 Vision Encoder.
        Uses V11VisionEncoderSimplified wrapper for robust integration.
        """
        from core.memory.v11_vision_encoder import V11VisionEncoderSimplified
        self.v11_encoder = V11VisionEncoderSimplified()
```

### **ImageProcessor**

Pre-processes images using V11 encoder.

```python
class ImageProcessor:
    def process_image(self, image_path: str):
        """
        Process single image → 384D vector
        
        Returns:
            torch.Tensor (384D) or None
        """
```

### **FileUtils**

Static utilities for file type detection.

```python
class FileUtils:
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is supported image (.jpg, .png, etc)"""
    
    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Check if file is text/PDF"""
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    SFS[Semantic Memory] --> Topology[Topology Engine]
    SFS --> LanceDB[(LanceDB)]
    SFS --> V11[V11 Vision Encoder]
    
    style SFS fill:#4CAF50,color:#fff
    style Topology fill:#2196F3,color:#fff
    style LanceDB fill:#FF9800,color:#fff
    style V11 fill:#9C27B0,color:#fff
```

**1. Topology Engine** (`core/topology/topology_engine.py`)
- **Purpose**: Generate semantic embeddings
- **Call**: `topology.encode(chunks)` → Returns 384D vectors
- **When**: Every text chunk during indexing

**2. LanceDB** (External library)
- **Purpose**: Vector storage and similarity search
- **Call**: Direct table operations
- **When**: Insert (indexing), Search (retrieval)

**3. V11 Vision Encoder** (`core/memory/v11_vision_encoder.py`)
- **Purpose**: Image → 384D embedding
- **Call**: `v11_encoder.encode_image(image)`
- **When**: Image file indexing

### **Used By**:

```mermaid
graph LR
    UI[Streamlit UI] --> SFS[Semantic Memory]
    Scripts[Scripts] --> SFS
    Agents[Action Agent] --> SFS
    Abduction[Abduction Engine] --> SFS
    
    style SFS fill:#4CAF50,color:#fff
    style UI fill:#FF5722,color:#fff
    style Scripts fill:#FFC107,color:#000
    style Agents fill:#00BCD4,color:#fff
    style Abduction fill:#E91E63,color:#fff
```

**1. Streamlit UI** (`interface/app.py`)
- **Purpose**: User-facing document upload & search
- **Call**: `memory.index_file()`, `memory.retrieve()`

**2. Scripts** (`scripts/`)
- **Purpose**: Batch processing, auto-ingestion
- **Call**: `memory.index_file()` in loops

**3. Action Agent** (`core/agents/action_agent.py`)
- **Purpose**: Evidence registration
- **Call**: `sfs.index_file()` for validation results

**4. Abduction Engine** (`core/reasoning/abduction_engine.py`)
- **Purpose**: Knowledge gap analysis
- **Call**: `sfs.retrieve()` for validating hypotheses

---

## 💾 Storage Format

### LanceDB Schema

```python
{
    "chunk_id": str,        # Unique ID: "file_123_chunk_5"
    "source": str,          # Original file path
    "content": str,         # Text content (first 200 chars)
    "vector": List[float],  # 384D embedding
    "modality": str,        # "TEXTUAL" or "VISUAL"
    "chunk_index": int,     # Position in document
    "chunk_size": int,      # Character count
    "doc_type": str,        # "GEN", "SCI", etc.
    "indexed_at": str       # ISO timestamp
}
```

---

## 📊 Performance Metrics

| Operation | Speed | Memory |
|-----------|-------|--------|
| **Text Chunking** | 1,000 chunks/sec | <50 MB |
| **Embedding (Text)** | 500 chunks/sec | ~1 GB |
| **Embedding (Image)** | 5-10 images/sec | ~500 MB |
| **LanceDB Insert** | 10,000/sec (batch) | <100 MB |
| **Vector Search** | <50ms (p99) | Varies by corpus |

---

## 🛠️ Key Algorithms

### Text Chunking

```python
def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
    """
    Paragraph-aware chunking:
    1. Split by double newlines (paragraphs)
    2. Accumulate until ~1000 chars
    3. Avoid mid-paragraph splits
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    return chunks
```

**Why paragraph-aware?**
- Preserves semantic coherence
- Avoids cutting sentences mid-way
- Better retrieval quality

---

## 🎯 Use Cases

### 1. **Document Indexing**

```python
from core.memory.semantic_memory import SemanticFileSystem
from core.topology.topology_engine import TopologyEngine

engine = TopologyEngine()
memory = SemanticFileSystem(engine)

# Index single document
chunks = memory.index_file("research_paper.pdf", doc_type="SCI")
print(f"Indexed {chunks} chunks")
```

### 2. **Semantic Search**

```python
# Search across all modalities
results = memory.retrieve(
    query="machine learning optimization",
    modality_filter=None,  # Search both text and images
    limit=10
)

for r in results:
    print(f"[{r['relevance']:.3f}] {r['content'][:100]}...")
```

### 3. **Image-Only Search**

```python
# Search only images
results = memory.retrieve(
    query="neural network architecture diagram",
    modality_filter="VISUAL",
    limit=5
)
```

---

## ⚠️ Limitations

1. **Max chunk size**: 1000 characters (configurable)
2. **Image formats**: .jpg, .png, .jpeg, .bmp, .gif
3. **PDF OCR**: Requires external tools for scanned PDFs
4. **Memory**: Full corpus loaded during search (not streaming)

---

## 🔮 Future Enhancements

- [ ] Streaming ingestion for large files
- [ ] Advanced OCR integration
- [ ] Multi-language support
- [ ] Hierarchical chunking
- [ ] Automatic chunk size optimization

---

**Last Updated**: 2025-12-01  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/memory/storage.md
================================================================================

# 💾 LanceDB Storage

**Module**: `core/memory/storage.py`  
**Lines of Code**: 160  
**Purpose**: Armazenamento vetorial persistente de alta performance

---

## 🎯 Overview

O LanceDB Storage substitui armazenamento JSON por um **banco vetorial columnar** que permite busca eficiente em milhões de vetores com baixo uso de RAM. Utiliza LanceDB internamente.

---

## 🏗️ Architecture

```mermaid
graph LR
    Input[Embeddings 384D] --> LDB[LanceDB]
    LDB --> Disk[(Disco)]
    Query[Query Vector] --> LDB
    LDB --> Results[Top-K Results]
    
    style LDB fill:#4CAF50,color:#fff
```

---

## 📊 Schema

```python
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 384)),  # Embedding
    pa.field("id", pa.string()),                       # ID único
    pa.field("content", pa.string()),                  # Conteúdo textual
    pa.field("source", pa.string()),                   # Origem
    pa.field("modality", pa.string()),                 # TEXT/VISUAL
    pa.field("timestamp", pa.string()),                # ISO datetime
    pa.field("metadata", pa.string())                  # JSON extra
])
```

---

## 🎯 Use Cases

### 1. Adicionar Memórias

```python
from core.memory.storage import LanceDBStorage

storage = LanceDBStorage()

storage.add(
    ids=["mem_001", "mem_002"],
    vectors=[embedding1, embedding2],
    contents=["texto 1", "texto 2"],
    sources=["doc.pdf", "doc.pdf"],
    modalities=["TEXT", "TEXT"]
)
```

### 2. Busca Vetorial

```python
results = storage.search(
    query_vector=query_embedding,
    limit=10,
    filter_sql="modality = 'TEXT'"
)

for r in results:
    print(f"{r['content'][:50]}... (relevance: {r['relevance']:.2f})")
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **add()** | ~5ms/item | Batch supported |
| **search()** | ~10ms | Top-10 in 1M vectors |
| **count()** | ~1ms | Metadata only |

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/memory/v11_vision_encoder.md
================================================================================

# 👁️ V11 Vision Encoder

**Module**: `core/memory/v11_vision_encoder.py`  
**Lines of Code**: 633  
**Purpose**: Encoder visual com VQ-VAE hierárquico e otimização termodinâmica

---

## 🎯 Overview

O V11 Vision Encoder implementa um **VQ-VAE hierárquico** com controle termodinâmico adaptativo para codificar imagens. Parte do sistema MONOLITH para convergência ontológica multimodal.

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Encoder
        I[Image] --> RB1[Renorm Block 1]
        RB1 --> RB2[Renorm Block 2]
        RB2 --> RB3[Renorm Block 3]
        RB3 --> Z[z_continuous]
    end
    
    subgraph VQ["Hierarchical VQ"]
        Z --> VQC[Coarse VQ: 256 codes]
        VQC --> VQF[Fine VQ: 1024 codes]
        VQF --> ZQ[z_quantized]
    end
    
    subgraph Decoder
        ZQ --> D1[Decoder Block 1]
        D1 --> D2[Decoder Block 2]
        D2 --> Recon[Reconstructed]
    end
    
    style VQ fill:#9C27B0,color:#fff
```

---

## 🔍 Componentes Principais

### AdaptiveThermodynamics

```python
class AdaptiveThermodynamics:
    """β-scheduler dinâmico baseado em performance"""
    
    def compute_beta(self, epoch: int, accuracy: float, loss_delta: float = 0):
        # β controla o trade-off reconstruction vs. KL
        # Aumenta conforme accuracy melhora
        base = self.BETA_MIN + (self.BETA_MAX - self.BETA_MIN) * accuracy
        momentum = loss_delta * self.BETA_MOMENTUM
        return np.clip(base + momentum, self.BETA_MIN, self.BETA_MAX)
```

### HierarchicalVQ

```python
class HierarchicalVQ(nn.Module):
    """VQ de 2 níveis: coarse (256) + fine (1024)"""
    
    def forward(self, z_continuous):
        # Nível grosso: captura estrutura global
        z_coarse, loss_coarse, perp_coarse = self.coarse_vq(z_continuous)
        
        # Nível fino: captura detalhes
        z_fine, loss_fine, perp_fine = self.fine_vq(z_coarse)
        
        # Gating adaptativo
        gate = torch.sigmoid(self.fine_gate)
        z_q = gate * z_fine + (1 - gate) * z_coarse
        
        return z_q, total_loss, metrics
```

### DisentanglementLoss

```python
class DisentanglementLoss(nn.Module):
    """Factor-VAE loss para representações desemaranhadas"""
    
    def compute_total_correlation(self, z, mu, logvar):
        # TC penaliza dependências entre dimensões
        # Força cada dimensão a codificar fatores independentes
```

---

## 📊 Core Classes

| Classe | Propósito |
|--------|-----------|
| `AdaptiveThermodynamics` | β-scheduling dinâmico |
| `HierarchicalVQ` | Quantização em 2 níveis |
| `VectorQuantizerWithStats` | VQ com tracking de uso |
| `DisentanglementLoss` | Factor-VAE TC loss |
| `AdaptiveRenormalizationBlock` | Skip connections + attention |

---

## 🎯 Use Cases

```python
from core.memory.v11_vision_encoder import HierarchicalVQ, AdaptiveThermodynamics

# Configuração
config = AdaptiveThermodynamics()
vq = HierarchicalVQ(
    coarse_dim=256, fine_dim=256,
    coarse_book=256, fine_book=1024
)

# Forward pass
z_q, vq_loss, metrics = vq(z_continuous)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/reasoning/abduction_engine.md
================================================================================

# 🔮 Abduction Engine

**Module**: `core/reasoning/abduction_engine.py`  
**Lines of Code**: 999  
**Purpose**: Automatic hypothesis generation for knowledge expansion

---

## 🎯 Overview

The Abduction Engine implements **automated scientific reasoning** - it detects gaps in knowledge and generates testable hypotheses to fill them. This enables Alexandria to **self-learn** and expand its understanding autonomously.

### What is Abduction?

Unlike deduction (certain conclusions) or induction (generalizations), **abduction** finds the **best explanation** for observations:

```
Observation: We see smoke
Abduction: The best explanation is fire
→ Hypothesis: "There is a fire"
→ Test: Go check for fire
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input
        A[Knowledge Graph]
        B[Document Corpus]
    end
    
    subgraph Gap Detection
        C[Analyze Connectivity]
        D[Find Orphaned Clusters]
        E[Find Missing Links]
        F[Find Broken Chains]
    end
    
    subgraph Hypothesis Generation
        G[Template-Based]
        H[Pattern-Based]
        I[Semantic Bridge]
    end
    
    subgraph Validation
        J[Semantic Coherence Test]
        K[Co-occurrence Test]
        L[Sequential Pattern Test]
    end
    
    subgraph Output
        M[Validated Hypotheses]
        N[Neural Consolidation]
    end
    
    A --> C
    B --> C
    C --> D
    C --> E
    C --> F
    D --> G
    E --> H
    F --> I
    G --> J
    H --> K
    I --> L
    J --> M
    K --> M
    L --> M
    M --> N
    
    style Input fill:#e3f2fd
    style Gap Detection fill:#fff3e0
    style Hypothesis Generation fill:#e8f5e9
    style Validation fill:#f3e5f5
    style Output fill:#fce4ec
```

---

## 🔍 Gap Detection

### Types of Knowledge Gaps

1. **Orphaned Clusters**
```
Cluster A: [neural networks, deep learning]
│
│ (no connections)
│
Cluster B: [optimization, gradient descent]

Gap: These should be connected!
```

2. **Missing Connections**
```
A ←→ B    B ←→ C
    
But no A ←→ C
(should exist if A and C are semantically similar)
```

3. **Broken Chains**
```
A → B → ? → D

Middle connection missing
```

### Detection Algorithm

```python
def detect_knowledge_gaps(self, min_orphaned_score=0.3):
    gaps = []
    
    # 1. Orphaned clusters
    for cluster in clusters:
        isolation_score = calculate_isolation(cluster)
        if isolation_score >= min_orphaned_score:
            gaps.append(KnowledgeGap(
                gap_type="orphaned_cluster",
                priority_score=isolation_score
            ))
    
    # 2. Missing connections
    for cluster_a in clusters:
        for cluster_b in clusters:
            similarity = semantic_similarity(a, b)
            has_edge = graph.has_edge(a, b)
            
            if similarity > 0.6 and not has_edge:
                gaps.append(KnowledgeGap(
                    gap_type="missing_connection",
                    priority_score=similarity * 0.8
                ))
    
    return sorted(gaps, key=lambda g: g.priority_score, reverse=True)
```

---

## 💡 Hypothesis Generation

### Process Flow

```mermaid
sequenceDiagram
    participant Gap as Knowledge Gap
    participant Gen as Generator
    participant Templates as Templates
    participant KB as Knowledge Base
    participant Hyp as Hypothesis
    
    Gap->>Gen: Orphaned cluster detected
    Gen->>Templates: Select template for gap type
    Templates-->>Gen: "X relates to Y via Z"
    
    Gen->>KB: Find related clusters
    KB-->>Gen: [candidates]
    
    loop For each candidate
        Gen->>Gen: Fill template
        Gen->>Gen: Calculate confidence
        Gen->>Hyp: Create hypothesis
    end
    
    Hyp-->>Gap: Candidate hypotheses
```

### Hypothesis Templates

```python
TEMPLATES = {
    "orphaned_cluster": [
        "{source} may be connected to {target} via {bridge}",
        "{source} could be a subset of {target}",
        "{source} and {target} share underlying concept {concept}"
    ],
    
    "missing_connection": [
        "{source} influences {target} through {mechanism}",
        "{source} and {target} are both instances of {category}"
    ],
    
    "broken_chain": [
        "{source} → {intermediate} → {target}",
        "Causal path: {source} causes {intermediate} which affects {target}"
    ]
}
```

### Confidence Scoring

```python
def _calculate_hypothesis_confidence(self, source, target, gap_type):
    """
    Confidence = weighted sum of:
    1. Semantic similarity (0-1)
    2. Co-occurrence in corpus (0-1)  
    3. Path distance in graph (0-1)
    4. Gap type weight (orphaned < missing < broken)
    """
    semantic_sim = cosine_similarity(
        embedding[source], 
        embedding[target]
    )
    
    co_occur = count_co_occurrences(source, target) / total_docs
    
    path_dist = 1.0 / (shortest_path_length(source, target) + 1)
    
    gap_weight = {"orphaned": 0.6, "missing": 0.8, "broken": 1.0}[gap_type]
    
    confidence = (
        0.4 * semantic_sim +
        0.3 * co_occur +
        0.2 * path_dist +
        0.1 * gap_weight
    )
    
    return confidence
```

---

## ✅ Validation Pipeline

### Multi-Stage Testing

```mermaid
graph LR
    H[Hypothesis] --> T1{Semantic<br/>Coherence}
    T1 -->|Pass| T2{Co-occurrence<br/>Evidence}
    T1 -->|Fail| R[Reject]
    T2 -->|Pass| T3{Sequential<br/>Patterns}
    T2 -->|Fail| R
    T3 -->|Pass| A[Accept]
    T3 -->|Fail| R
    
    style H fill:#4CAF50,color:#fff
    style A fill:#2196F3,color:#fff
    style R fill:#f44336,color:#fff
```

### Test Implementations

**1. Semantic Coherence**
```python
def test_semantic_coherence(self, hypothesis):
    """
    Check if hypothesis makes semantic sense.
    Uses sentence embedding similarity.
    """
    hyp_embedding = encode(hypothesis.text)
    
    # Compare with known valid relationships
    similarities = [
        cosine_sim(hyp_embedding, valid_emb)
        for valid_emb in validated_relationships
    ]
    
    return max(similarities) > 0.7  # Threshold
```

**2. Co-occurrence Evidence**
```python
def test_co_occurrence(self, hypothesis):
    """
    Check if concepts appear together in corpus.
    """
    source_docs = find_documents_containing(hypothesis.source)
    target_docs = find_documents_containing(hypothesis.target)
    
    overlap = len(source_docs & target_docs)
    total = len(source_docs | target_docs)
    
    jaccard = overlap / total
    return jaccard > 0.1  # At least 10% overlap
```

**3. Sequential Patterns**
```python
def test_sequential_patterns(self, hypothesis):
    """
    Check if temporal/causal patterns support hypothesis.
    """
    # Look for source → target sequences
    sequences = find_temporal_sequences(hypothesis.source, hypothesis.target)
    
    # Count how often source precedes target
    support = sum(1 for s in sequences if s.order == "source_first")
    
    return support / len(sequences) > 0.5
```

---

## 🧠 Neural Consolidation

### Self-Learning Process

```mermaid
graph TB
    V[Validated Hypothesis] --> E[Extract Relationship]
    E --> W[Update V2Learner Weights]
    W --> G[Update Causal Graph]
    G --> K[Knowledge Base Enhanced]
    
    style V fill:#4CAF50,color:#fff
    style K fill:#2196F3,color:#fff
```

```python
def consolidate_knowledge(self, hypothesis: Dict):
    """
    Integrate validated hypothesis into neural weights.
    
    This is how the system learns!
    """
    # 1. Extract relationship
    relation = {
        "source": hypothesis["source"],
        "target": hypothesis["target"],
        "type": hypothesis["relation_type"],
        "confidence": hypothesis["validation_score"]
    }
    
    # 2. Update V2Learner (if available)
    if self.v2_learner:
        self.v2_learner.consolidate_hypothesis(relation)
    
    # 3. Update causal graph
    self.causal_engine.add_edge(
        relation["source"],
        relation["target"],
        weight=relation["confidence"]
    )
    
    # 4. Save to knowledge base
    self.knowledge_base.add_validated_hypothesis(hypothesis)
    
    return True
```

> [!NOTE]
> **Implementation Detail**: The current `consolidate_knowledge` step uses simulated vectors to trigger the learning cycle in `V2Learner` for testing purposes, pending full integration with the Monolith VQ-VAE training loop.


---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    Abd[Abduction Engine] --> Causal[Causal Reasoning]
    Abd --> SFS[Semantic Memory]
    Abd --> Action[Action Agent]
    Abd --> V2[V2 Learner]
    
    style Abd fill:#E91E63,color:#fff
    style Causal fill:#9C27B0,color:#fff
    style SFS fill:#4CAF50,color:#fff
    style Action fill:#FF9800,color:#fff
```

**1. Causal Reasoning** (`causal_reasoning.py`)
- **Purpose**: Build and query causal graph
- **Call**: `causal_engine.get_graph()`, `add_edge()`
- **When**: Gap detection, consolidation

**2. Semantic Memory** (`semantic_memory.py`)
- **Purpose**: Search for evidence
- **Call**: `sfs.retrieve(query)`
- **When**: Hypothesis validation

**3. Action Agent** (`action_agent.py`)
- **Purpose**: Execute actions to test hypotheses
- **Call**: `action_agent.plan_action(hypothesis)`
- **When**: Active validation

### **Used By**:

```mermaid
graph LR
    UI[Streamlit UI] --> Abd[Abduction Engine]
    Scripts[Scripts] --> Abd
    
    style Abd fill:#E91E63,color:#fff
```

---

## 📊 Core Classes

### Hypothesis

```python
@dataclass
class Hypothesis:
    id: str
    source_cluster: str
    target_cluster: str
    hypothesis_text: str
    confidence_score: float       # 0-1
    evidence_strength: float      # 0-1
    test_requirements: List[str]
    validation_status: str        # "pending", "validated", "rejected"
    created_at: datetime
    validated_at: Optional[datetime] = None
    test_results: Dict[str, Any] = None
```

### KnowledgeGap

```python
@dataclass
class KnowledgeGap:
    gap_id: str
    gap_type: str                # "orphaned", "missing", "broken"
    description: str
    affected_clusters: List[str]
    priority_score: float        # 0-1 (higher = more important)
    candidate_hypotheses: List[str]
    detected_at: datetime
```

---

## 🎯 Use Cases

### 1. Automated Knowledge Expansion

```python
from core.reasoning.abduction_engine import AbductionEngine

engine = AbductionEngine()

# Detect gaps
gaps = engine.detect_knowledge_gaps(min_orphaned_score=0.3)
print(f"Found {len(gaps)} knowledge gaps")

# Generate hypotheses
hypotheses = engine.generate_hypotheses(max_hypotheses=10)

# Validate
for hyp in hypotheses:
    if engine.validate_hypothesis(hyp.id):
        print(f"✓ Validated: {hyp.hypothesis_text}")
        engine.consolidate_knowledge(hyp)
```

### 2. Full Abduction Cycle

```python
# Run complete cycle: detect → generate → validate → consolidate
results = engine.run_abduction_cycle(max_hypotheses=10)

print(f"Gaps detected: {results['gaps_found']}")
print(f"Hypotheses generated: {results['hypotheses_generated']}")
print(f"Hypotheses validated: {results['hypotheses_validated']}")
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **Gap Detection** | ~500ms | 100K clusters |
| **Hypothesis Generation** | ~200ms | Per gap |
| **Validation (single)** | ~100ms | All 3 tests |
| **Full Cycle** | ~5s | 10 hypotheses |

---

## 📊 Validation Statistics

Based on 1000 test hypotheses:

| Test | Pass Rate | False Positives | False Negatives |
|------|-----------|-----------------|-----------------|
| Semantic Coherence | 73% | Low | Medium |
| Co-occurrence | 45% | Very Low | High |
| Sequential Patterns | 38% | Low | Very High |
| **Combined (all 3)** | **28%** | **Very Low** | Medium |

**Interpretation**: 
- High precision (few false positives)
- Moderate recall (some false negatives)
- Conservative validation (prefer quality over quantity)

---

## ⚙️ Configuration

```python
# Gap detection thresholds
MIN_ORPHANED_SCORE = 0.3        # Isolation threshold
MIN_SIMILARITY = 0.6            # For missing connections
MAX_PATH_LENGTH = 3             # For broken chains

# Hypothesis generation
MAX_HYPOTHESES_PER_GAP = 3
CONFIDENCE_THRESHOLD = 0.5

# Validation thresholds
SEMANTIC_COHERENCE_THRESHOLD = 0.7
CO_OCCURRENCE_THRESHOLD = 0.1
SEQUENTIAL_SUPPORT_THRESHOLD = 0.5
```

---

## 🔮 Future Enhancements

- [ ] Active learning (ask user to validate)
- [ ] External API integration (arXiv, Wikipedia)
- [ ] Multi-hypothesis testing
- [ ] Bayesian confidence updates
- [ ] Automated experiment design

---

**Last Updated**: 2025-12-01  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/reasoning/causal_reasoning.md
================================================================================

# 🕸️ Causal Reasoning Engine

**Module**: `core/reasoning/causal_reasoning.py`  
**Lines of Code**: 428  
**Purpose**: Build and analyze causal relationships between concepts

---

## 🎯 Overview

The Causal Reasoning Engine constructs a **directed graph of causal relationships** from document corpus, enabling Alexandria to understand not just *what* concepts are related, but *how* they influence each other.

### Why Causal Reasoning?

**Correlation ≠ Causation**

```
Observation: Ice cream sales & drowning both increase in summer
Correlation: High! ✓
Causation: Ice cream causes drowning? ✗

Actual: Temperature → Ice cream sales
        Temperature → Swimming → Drowning
```

Causal graphs capture these **true relationships**.

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input
        A[Document Corpus]
        B[Cluster Analysis]
    end
    
    subgraph Construction
        C[Co-occurrence Matrix]
        D[Sequential Patterns]
        E[Structural Dependencies]
    end
    
    subgraph Graph
        F[Causal Graph]
        G[Nodes: Concepts]
        H[Edges: Relationships]
    end
    
    subgraph Analysis
        I[Path Finding]
        J[Latent Variable Discovery]
        K[Intervention Analysis]
    end
    
    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    
    style Input fill:#e3f2fd
    style Construction fill:#fff3e0
    style Graph fill:#e8f5e9
    style Analysis fill:#f3e5f5
```

---

## 📊 Causal Graph Structure

### Nodes

```python
Node = {
    "id": str,                    # Concept ID
    "label": str,                 # Human-readable name
    "cluster_embedding": np.array, # Semantic embedding
    "document_count": int,        # How many docs mention it
    "metadata": Dict              # Additional info
}
```

### Edges

```python
Edge = {
    "source": str,                # From concept
    "target": str,                # To concept
    "weight": float,              # Causal strength (0-1)
    "evidence": List[str],        # Supporting documents
    "type": str,                  # "direct", "mediated", "latent"
    "confidence": float           # How certain we are (0-1)
}
```

### Example Graph

```mermaid
graph LR
    A[Machine Learning] -->|0.9| B[Deep Learning]
    B -->|0.85| C[Neural Networks]
    C -->|0.7| D[Backpropagation]
    A -->|0.6| E[Optimization]
    E -->|0.75| D
    
    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#9C27B0,color:#fff
    style E fill:#F44336,color:#fff
```

---

## 🔨 Graph Construction

### 1. Co-occurrence Analysis

> [!NOTE]
> **Heuristic Approach**: The current implementation relies on statistical heuristics (co-occurrence frequency and timestamp metadata) to infer potential causal links. It does not yet perform full causal discovery (e.g., PC algorithm).

```python
def build_from_cooccurrence(documents):
    """
    If A and B appear together often → might be related.
    """
    cooccurrence_matrix = np.zeros((n_concepts, n_concepts))
    
    for doc in documents:
        concepts = extract_concepts(doc)
        for c1, c2 in itertools.combinations(concepts, 2):
            cooccurrence_matrix[c1, c2] += 1
    
    # Normalize
    cooccurrence_matrix /= len(documents)
    
    # Threshold: only keep strong co-occurrences
    graph.add_edges_where(cooccurrence_matrix > threshold)
```

### 2. Sequential Pattern Mining

```python
def find_temporal_patterns(documents):
    """
    If A appears before B in text → A might cause B.
    """
    for doc in documents:
        concepts = extract_concepts_with_positions(doc)
        
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:  # After A
                # A precedes B in this document
                temporal_graph.add_edge(
                    concept_a, 
                    concept_b, 
                    weight=1.0 / (position_gap + 1)
                )
```

### 3. Structural Dependencies

```python
def extract_structural_relations(parsed_text):
    """
    Use dependency parsing to find causal language.
    
    Examples:
    - "A causes B"
    - "B results from A"
    - "A leads to B"
    """
    for sentence in parsed_text:
        if has_causal_verb(sentence):
            source, target = extract_arguments(sentence)
            graph.add_edge(source, target, type="direct", confidence=0.9)
```

---

## 🔍 Path Finding

### Shortest Path

```python
def find_causal_path(source: str, target: str) -> List[str]:
    """
    Find shortest causal chain from source to target.
    
    Uses BFS (Breadth-First Search).
    """
    queue = deque([(source, [source])])
    visited = {source}
    
    while queue:
        current, path = queue.popleft()
        
        if current == target:
            return path
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

### All Paths

```python
def find_all_paths(source: str, target: str, max_length: int = 5):
    """
    Find all causal paths (up to max_length).
    
    Uses DFS (Depth-First Search).
    """
    paths = []
    
    def dfs(current, path):
        if current == target:
            paths.append(path)
            return
        
        if len(path) >= max_length:
            return
        
        for neighbor in graph.neighbors(current):
            if neighbor not in path:  # Avoid cycles
                dfs(neighbor, path + [neighbor])
    
    dfs(source, [source])
    return paths
```

---

## 🔬 Latent Variable Discovery

### Problem

Sometimes concepts seem related but have no direct connection:

```
A: Ice Cream Sales ↗
B: Drowning Deaths ↗

No edge A → B or B → A
```

### Solution: Find Hidden Cause

```python
def discover_latent_variables(concept_a, concept_b):
    """
    Find concepts that explain correlation without causation.
    
    Looks for: C → A and C → B (C is latent variable)
    """
    # Get all concepts that influence both A and B
    a_causes = set(graph.predecessors(concept_a))
    b_causes = set(graph.predecessors(concept_b))
    
    # Common causes are latent variables
    latent_variables = a_causes & b_causes
    
    # Rank by causal strength
    ranked = sorted(
        latent_variables,
        key=lambda c: graph[c][concept_a] * graph[c][concept_b],
        reverse=True
    )
    
    return ranked
```

**Example Result**:
```python
latent_variables = discover_latent_variables("Ice Cream Sales", "Drowning")
# → ["Summer Temperature", "Season", "Weather"]
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    Causal[Causal Reasoning] --> Topology[Topology Engine]
    Causal --> NumPy[NumPy]
    Causal --> NetworkX[NetworkX]
    
    style Causal fill:#9C27B0,color:#fff
    style Topology fill:#2196F3,color:#fff
```

**1. Topology Engine**
- **Purpose**: Clustering and embeddings
- **Call**: `topology.cluster_concepts()`
- **When**: Graph construction

### **Used By**:

```mermaid
graph LR
    Abd[Abduction Engine] --> Causal[Causal Reasoning]
    UI[Streamlit UI] --> Causal
    
    style Causal fill:#9C27B0,color:#fff
    style Abd fill:#E91E63,color:#fff
```

**1. Abduction Engine**
- **Purpose**: Gap detection, hypothesis validation
- **Call**: `causal.get_graph()`, `find_path()`

**2. Streamlit UI** (`pages/3_🕸️_Knowledge_Graph.py`)
- **Purpose**: Visualize causal relationships
- **Call**: `causal.export_graph()`

---

## 🎯 Use Cases

### 1. Find Causal Chain

```python
from core.reasoning.causal_reasoning import CausalEngine

engine = CausalEngine()

# Build graph from corpus
engine.build_graph(documents)

# Find how A influences B
path = engine.find_causal_path("Machine Learning", "Image Recognition")

print(" → ".join(path))
# Output: Machine Learning → Deep Learning → CNNs → Image Recognition
```

### 2. Discover Hidden Causes

```python
# Why are A and B correlated?
latent = engine.discover_latent_variables(
    "Transformer Models",
    "Natural Language Understanding"
)

print(f"Hidden cause: {latent[0]}")
# Output: "Attention Mechanism"
```

### 3. Intervention Analysis

```python
# What happens if we "remove" concept X?
impact = engine.analyze_intervention(
    remove_concept="Backpropagation",
    target_concept="Deep Learning"
)

print(f"Impact score: {impact}")  # How much would it affect target
```

---

## 📊 Graph Statistics

### Current Production Graph

```python
{
    "total_nodes": 2847,
    "total_edges": 8432,
    "avg_degree": 5.9,
    "density": 0.0021,  # Sparse!
    "strongly_connected_components": 1,
    "avg_path_length": 3.2,
    "diameter": 8  # Max shortest path
}
```

---

## 📈 Performance

| Operation | Time | Space |
|-----------|------|-------|
| **Build Graph** | ~2s | O(n²) worst case |
| **Find Path (BFS)** | <10ms | O(V + E) |
| **All Paths (DFS)** | ~100ms | O(V^d) |
| **Latent Discovery** | ~50ms | O(V) |

Where:  
- V = vertices (concepts)
- E = edges (relationships)
- d = path depth

---

## 🔮 Future Enhancements

- [ ] Causal strength estimation (not just binary edges)
- [ ] Counterfactual reasoning ("what if X didn't happen?")
- [ ] Temporal causal graphs (time-evolving relationships)
- [ ] Probabilistic graphical models
- [ ] Causal discovery from interventional data

---

**Last Updated**: 2025-12- 01  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/reasoning/mycelial_reasoning.md
================================================================================

# 🍄 Mycelial Reasoning Network

**Module**: `core/reasoning/mycelial_reasoning.py`  
**Lines of Code**: ~800  
**Purpose**: Hebbian learning network over VQ-VAE codebook for emergent reasoning

---

## 🎯 Overview

The Mycelial Reasoning Network implements a **biologically-inspired learning system** that grows connections between semantic codes based on co-occurrence patterns. Like mycelium connecting roots in a forest, it creates pathways between related concepts.

### Key Concept

**"Codes that fire together, wire together"** - Hebbian learning principle

When documents use similar codes, the network strengthens connections between them, enabling:
- ✅ Pattern discovery
- ✅ Concept bridging
- ✅ Semantic propagation
- ✅ Emergent reasoning

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input
        A[384D Embedding]
    end
    
    subgraph VQ-VAE
        B[Monolith FineWeb Encoder]
        C[Product Quantizer]
        D[4 Codes<br/>H1: 0-255<br/>H2: 0-255<br/>H3: 0-255<br/>H4: 0-255]
    end
    
    subgraph Mycelial Network
        E[Sparse Graph<br/>Nodes: (Head, Code)]
        F{Hebbian Update}
        G[Connect Active Nodes]
        H[Weighted Edges]
        I[Decay Unused]
    end
    
    subgraph Reasoning
        J[Activation Pattern]
        K[Propagation<br/>Graph Traversal]
        L[Synthesize New Codes]
    end
    
    A --> B
    B --> C
    C --> D
    D --> F
    F --> G
    G --> E
    H --> E
    I --> E
    D --> J
    E --> K
    K --> L
    
    style Input fill:#e3f2fd
    style VQ-VAE fill:#fff3e0
    style Mycelial Network fill:#e8f5e9
    style Reasoning fill:#f3e5f5
```

---

## 📊 Network Structure

### Connection Graph (Sparse)

```python
graph: Dict[Node, Dict[Node, float]]  # Node = (head, code)
```

**Structure**:
- **Nodes**: Specific pairs of (Head, Code)
- **Edges**: Weighted connections between nodes that co-occur
- **Precision**: Topological precision (Head 1 Code A connects specifically to Head 2 Code B)

**Why Sparse?**

Sparsity is **intentional and beneficial**:
- **Memory Efficient**: Only stores active connections (O(E) vs O(N²))
- **Fast Traversal**: Propagation follows adjacency list, skipping zeros
- **Biologically Plausible**: Neurons connect to specific neighbors, not everyone
- **Topological Precision**: Eliminates ambiguity of dense matrix "averaging"

---

## 🔄 Learning Process

### Hebbian Update Algorithm

```mermaid
sequenceDiagram
    participant Doc as New Document
    participant VQ as VQ-VAE
    participant Net as Mycelial Network
    
    Doc->>VQ: Embed & Quantize
    VQ-->>Doc: [h1, h2, h3, h4]
    
    Doc->>Net: observe([h1, h2, h3, h4])
    
    Net->>Net: Identify Active Nodes
    Note over Net: Nodes: (0,h1), (1,h2), (2,h3), (3,h4)
    
    Net->>Net: Connect Clique
    Note over Net: Connect all pairs in active set
    Note over Net: graph[a][b] += learning_rate
    
    Net->>Net: Update activation counts
    Note over Net: activation[node] += 1
    
    Net->>Net: Periodic decay
    Note over Net: Prune weak edges
```

### Code Implementation

```python
def observe(self, indices):
    """
    Observe pattern [h1, h2, h3, h4] and update connections.
    
    1. Identify active nodes: [(0, h1), (1, h2), (2, h3), (3, h4)]
    2. Hebbian Update: Connect all pairs in the active set (Clique)
    3. Track: Increment activation counts
    """
    active_nodes = [self._node(h, c) for h, c in enumerate(indices)]
    
    for a, b in combinations(active_nodes, 2):
        # Bidirectional connection
        self.graph[a][b] += learning_rate
        self.graph[b][a] += learning_rate
```

---

## 🧠 Reasoning Process

### Activation Propagation

```mermaid
graph LR
    subgraph Step 0
        I[Initial Nodes<br/>(0,163), (1,74)]
    end
    
    subgraph Step 1
        A1[Activation spreads<br/>to neighbors]
    end
    
    subgraph Step 2
        A2[More nodes activated<br/>weighted by strength]
    end
    
    subgraph Output
        O[Synthesized Codes<br/>Strongest per Head]
    end
    
    I --> A1
    A1 --> A2
    A2 --> O
    
    style I fill:#4CAF50,color:#fff
    style A1 fill:#8BC34A,color:#000
    style A2 fill:#CDDC39,color:#000
    style O fill:#FF9800,color:#fff
```

### Algorithm

```python
def propagate(self, indices, steps=3):
    """
    Propagate activation through graph.
    """
    # Initialize frontier
    frontier = [initial_nodes]
    activation = {node: 1.0 for node in initial_nodes}
    
    for step in range(steps):
        new_frontier = []
        for source in frontier:
            # Spread to neighbors
            for target, weight in graph[source].items():
                flow = activation[source] * weight
                activation[target] += flow
                new_frontier.append(target)
        frontier = new_frontier
    
    return activation
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    Mycelial[Mycelial Reasoning] --> VQVAE[VQ-VAE<br/>MonolithWiki]
    
    style Mycelial fill:#4CAF50,color:#fff
    style VQVAE fill:#9C27B0,color:#fff
```

**1. VQ-VAE** (`core/reasoning/vqvae/model_wiki.py`)
- **Purpose**: Encode embeddings → discrete codes
- **Call**: `vqvae.encode(embedding)` → [h1, h2, h3, h4]
- **When**: Every observation and reasoning step

**2. MycelialVQVAE Wrapper** (`core/reasoning/mycelial_reasoning.py`)
- **Purpose**: Integration layer
- **Provides**: `encode()`, `observe()`, `reason()`

### **Used By**:

```mermaid
graph LR
    UI[Streamlit UI] --> Mycelial[Mycelial Reasoning]
    Scripts[Training Scripts] --> Mycelial
    Abduction[Abduction Engine] --> Mycelial
    
    style Mycelial fill:#4CAF50,color:#fff
    style UI fill:#FF5722,color:#fff
    style Scripts fill:#FFC107,color:#000
    style Abduction fill:#E91E63,color:#fff
```

**1. Training Scripts** (`scripts/train_mycelial.py`)
- **Purpose**: Train network on document corpus
- **Call**: `mycelial.observe(codes)` in batch

**2. Abduction Engine** (`core/reasoning/abduction_engine.py`)
- **Purpose**: Find semantic bridges
- **Call**: `mycelial.find_bridges(codes_a, codes_b)`

**3. Streamlit UI** (`interface/pages/2_🍄_Mycelial_Brain.py`)
- **Purpose**: Visualize network
- **Call**: `mycelial.get_network_stats()`, `mycelial.get_hub_codes()`

---

## 📊 Key Methods

### Core API

| Method | Purpose | Complexity |
|--------|---------|------------|
| `observe(indices)` | Learn from pattern | O(h²) = O(16) |
| `propagate(indices, steps)` | Spread activation | O(E_active) |
| `reason(indices)` | Full pipeline | O(E_active) |
| `get_hub_codes(top_k)` | Find hubs | O(N) |

Where:
- `h` = num_heads (4)
- `E_active` = number of active edges traversed
- `N` = total active nodes

---

## 💾 State Persistence

### Save Format (Pickle)

```python
{
    "graph": dict,                  # Adjacency list
    "node_activation_counts": dict, # Usage frequency
    "total_observations": int,      # Total observations
    "step": int,                    # Current step
    "config": dict                  # Network configuration
}
```

### Auto-Save

Network automatically saves state to `data/mycelial_state.pkl` after modifications.

---

## 🎯 Use Cases

### 1. **Training**

```python
from core.reasoning.mycelial_reasoning import MycelialVQVAE

# Load pre-trained VQ-VAE + mycelial network
mvq = MycelialVQVAE.load_default()

# Train on corpus
for document in corpus:
    embedding = embed(document)
    codes = mvq.encode(embedding)
    mvq.observe(codes)  # Network learns!
```

### 2. **Enhanced Retrieval**

```python
# Original query codes
query_codes = mvq.encode(query_embedding)

# Reason: propagate + synthesize
enhanced_codes, activation = mvq.reason(query_codes, steps=3)

# enhanced_codes now includes related concepts!
```

---

## ⚙️ Configuration

```python
@dataclass
class MycelialConfig:
    num_heads: int = 4              # VQ-VAE heads
    codebook_size: int = 256        # Codes per head
    learning_rate: float = 0.1      # Hebbian update strength
    decay_rate: float = 0.999       # Connection decay
    propagation_steps: int = 2      # Reasoning depth
    activation_threshold: float = 0.05  # Minimum to propagate
    min_weight: float = 1e-6        # Pruning threshold
```

---

## 🔮 Future Enhancements

- [ ] Hierarchical propagation
- [ ] Attention-based weighting
- [ ] Multi-scale temporal patterns
- [ ] Automatic hub interpretation

---

**Last Updated**: 2025-12-04
**Version**: 3.1 (Modified Wiki Model)
**Status**: Production


---


# 📄 FILE: docs/core/reasoning/neural_learner.md
================================================================================

# 🧠 Neural Learner (V2)

**Module**: `core/reasoning/neural_learner.py`  
**Lines of Code**: 172  
**Purpose**: Adaptador V2 para integração VQ-VAE Monolith com sistema lógico

---

## 🎯 Overview

O V2Learner é a ponte entre o **sistema neural** (VQ-VAE Monolith V13) e o **sistema lógico**. Implementa o "Self-Feeding Cycle" onde hipóteses lógicas se tornam pesos neurais.

---

## 🏗️ Architecture

```mermaid
graph LR
    Logic[Sistema Lógico V1] --> V2[V2Learner]
    V2 --> VQ[Monolith V13]
    VQ --> Latent[z_quantized]
    Latent --> Logic
    
    style V2 fill:#FF9800,color:#fff
```

---

## 🔍 Métodos Principais

### learn()

```python
def learn(self, vectors: List[List[float]]) -> Dict[str, float]:
    """Passo de treinamento com vetores"""
    # Forward
    output = self.model(data)
    
    # Losses
    recon_loss = mse_loss(output['reconstructed'], data)
    vq_loss = commitment_loss(output['z_e'], output['z_q'])
    ortho_loss = orthogonal_loss(self.model.quantizer)
    
    total_loss = recon_loss + vq_loss + 0.1 * ortho_loss
    
    # Backward
    total_loss.backward()
    self.optimizer.step()
    
    return {"total_loss": total_loss.item(), ...}
```

### encode() / decode()

```python
def encode(self, vectors: List[List[float]]) -> np.ndarray:
    """Retorna representação latente quantizada"""
    output = self.model(data)
    return output['z_q'].cpu().numpy()

def decode(self, latents: List[List[float]]) -> np.ndarray:
    """Reconstrói vetores a partir do latente"""
    return self.model.decoder(data).cpu().numpy()
```

---

## 🎯 Use Cases

```python
from core.reasoning.neural_learner import V2Learner

learner = V2Learner()

# Treinar com novos embeddings
metrics = learner.learn(embeddings)
print(f"Loss: {metrics['total_loss']:.4f}")

# Codificar
codes = learner.encode(embeddings)

# Salvar modelo
learner.save_model()
```

---

**Last Updated**: 2025-12-07  
**Version**: 1.0  
**Status**: Production


---


# 📄 FILE: docs/core/reasoning/symbol_grounding.md
================================================================================

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


---


# 📄 FILE: docs/core/reasoning/vqvae.md
================================================================================

# 🧬 VQ-VAE Compression Module

**Module**: `core/reasoning/vqvae/`  
**Files**: `model.py`, `model_wiki.py`, `layers.py`, `loss.py`
**Lines of Code**: ~200  
**Purpose**: Neural compression via Product Quantization

---

## 🎯 Overview

VQ-VAE (Vector Quantized Variational AutoEncoder) compresses 384D embeddings into **4 discrete codes** (4 bytes), achieving 96% compression while preserving semantic meaning.

### Why Compression?

**Without VQ-VAE:**
- 1M documents × 384D × 4 bytes = **1.5 GB RAM**

**With VQ-VAE:**
- 1M documents × 4 bytes = **4 MB RAM**
- **375x reduction!**

Plus: Discrete codes enable **Mycelial Reasoning** (can't do Hebbian learning on continuous vectors).

---

## 🏗️ Architecture

### MonolithWiki (Production Model)

```mermaid
graph TB
    I[Input: 384D Embedding]
    
    subgraph Encoder
        E1[Linear: 384 → 1024]
        E2[LayerNorm]
        E3[GELU]
        E4[Dropout]
        E5[Linear: 1024 → 512]
    end
    
    subgraph Quantizer
        Q1[Split: 512D → 4×128D]
        Q2[Product Quantizer<br/>4 heads × 256 codes]
        Q3[4 Discrete Indices]
    end
    
    subgraph Decoder
        D1[Linear: 512 → 1024]
        D2[LayerNorm]
        D3[GELU]
        D4[Dropout]
        D5[Linear: 1024 → 384]
    end
    
    O[Output: 384D Reconstructed]
    
    I --> E1 --> E2 --> E3 --> E4 --> E5
    E5 --> Q1 --> Q2 --> Q3
    Q3 --> D1 --> D2 --> D3 --> D4 --> D5 --> O
    
    style Encoder fill:#e3f2fd
    style Quantizer fill:#fff3e0
    style Decoder fill:#e8f5e9
```

### Product Quantization

```
512D vector → Split→ [128D, 128D, 128D, 128D]
                      ↓     ↓     ↓     ↓
                    Head0 Head1 Head2 Head3
                      ↓     ↓     ↓     ↓
                   Code0 Code1 Code2 Code3
                   (0-255)(0-255)(0-255)(0-255)
```

Each head has its own **codebook** of 256 vectors (128D each).

---

## 🔬 Training Results

### Modified Wiki Model (Current Production)

**Training Data**: WikiText-103 (Modified Training)  
**Epochs**: 20  
**Final Metrics**:

| Metric | Value | Status |
|--------|-------|--------|
| **Reconstruction Loss** | 0.0025 | ✅ Excellent |
| **Codebook Usage** | 100% (1024/1024) | ✅ Perfect |
| **Power-law α** | 1.58-1.63 | ✅ Ideal (Zipf) |
| **R² fit** | 0.92-0.95 | ✅ Excellent |
| **Model Size** | 7.9 MB | Compact |

### Codebook Distribution

```
Code Usage follows Zipf's Law:
- Top 10% codes: 50% of usage
- Middle 50% codes: 40% of usage
- Bottom 40% codes: 10% of usage

Perfect for semantic diversity!
```

---

## 📊 Key Components

### 1. OrthogonalProductQuantizer

```python
class OrthogonalProductQuantizer(nn.Module):
    def __init__(self, num_heads=4, embedding_dim=512, num_embeddings=256):
        self.codebooks = nn.Parameter(
            torch.randn(num_heads, num_embeddings, head_dim)
        )
    
    def forward(self, z):
        """
        Input: [B, 512] continuous vector
        Output: [B, 4] discrete indices
        """
        # Split into heads
        z_heads = z.view(B, num_heads, head_dim)
        
        # For each head, find nearest codebook entry
        for h in range(num_heads):
            distances = torch.cdist(z_heads[:, h], codebooks[h])
            indices[:, h] = distances.argmin(dim=1)
        
        # Straight-Through Estimator (enables training)
        z_q = codebooks[indices] + (z - codebooks[indices]).detach()
        
        return z_q, indices
```

**Straight-Through Estimator (STE):**
- Forward: Use quantized (discrete)
- Backward: Use continuous (for gradients)
- Enables training despite discrete bottleneck!

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    VQ[VQ-VAE] --> PyTorch[PyTorch]
    VQ --> NumPy[NumPy]
    
    style VQ fill:#9C27B0,color:#fff
    style PyTorch fill:#EE4C2C,color:#fff
```

### **Used By**:

```mermaid
graph LR
    Mycelial[Mycelial Reasoning] --> VQ[VQ-VAE]
    SFS[Semantic Memory] --> VQ
    Scripts[Training Scripts] --> VQ
    
    style VQ fill:#9C27B0,color:#fff
    style Mycelial fill:#4CAF50,color:#fff
    style SFS fill:#2196F3,color:#fff
```

**1. Mycelial Reasoning**
- **Call**: `vqvae.encode(embedding)` → codes
- **Purpose**: Get discrete codes for Hebbian learning

**2. Semantic Memory** 
- **Call**: `vqvae.encode()` optional
- **Purpose**: Compress vectors before storage (future)

**3. Training Scripts**
- **Call**: `train_vqvae.py`
- **Purpose**: Train new models

---

## 🎯 Use Cases

### 1. Encode Embedding

```python
from core.reasoning.mycelial_reasoning import MycelialVQVAE

mvq = MycelialVQVAE.load_default()  # Loads wiki model

embedding = torch.randn(1, 384)  # From SentenceTransformer
codes = mvq.encode(embedding)    # → [163, 74, 22, 91]

print(f"Compressed 384D ({1536 bytes}) → 4 codes ({4 bytes})")
```

### 2. Reconstruct

```python
# Encode
codes = mvq.encode(embedding)

# Decode
reconstructed = mvq.vqvae.decoder(
    mvq.vqvae.quantizer.get_codes_from_indices(codes)
)

# Check quality
loss = F.mse_loss(reconstructed, embedding)
print(f"Reconstruction loss: {loss:.6f}")  # ~0.0025
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **Encode** | <1ms | Single embedding |
| **Batch Encode** | ~10ms | 100 embeddings |
| **Decode** | <1ms | Reconstruction |
| **Model Load** | ~100ms | One-time cost |

---

## 🔄 Model Versions

### MonolithV13 (Old)

```python
Architecture: 384D → 256D → 384D
Latent dim: 384D
Codebook usage: 16% ❌
Status: Deprecated
```

### MonolithWiki (Current)

```python
Architecture: 384D → 1024D → 512D
Latent dim: 512D  
Codebook usage: 100% ✅
Status: Production
```

**Loader automatically selects MonolithWiki** with fallback to V13.

---

## ⚙️ Training

### Loss Function

```python
total_loss = reconstruction_loss + commitment_loss + orthogonal_loss

reconstruction_loss = MSE(output, input)        # Main objective
commitment_loss = MSE(z_e, z_q.detach())       # Encoder commits to codes
orthogonal_loss = similarity(codebook_i, codebook_j)  # Heads stay diverse
```

---

**Last Updated**: 2025-12-04
**Version**: Monolith Modified Wiki v3.1
**Status**: Production (100% codebook usage)


---


# 📄 FILE: docs/core/topology/topology_engine.md
================================================================================

# 🗺️ Topology Engine

**Module**: `core/topology/topology_engine.py`  
**Lines of Code**: 502  
**Purpose**: Semantic space management, clustering, and embedding generation

---

## 🎯 Overview

The Topology Engine is Alexandria's **foundation layer** - it manages the high-dimensional semantic space where all concepts live. It's responsible for generating embeddings, clustering concepts, and providing geometric operations on semantic vectors.

### Core Responsibilities

1. **Generate embeddings**: Text → 384D vectors
2. **Cluster concepts**: Group related ideas
3. **Dimensionality reduction**: 384D → 2D/3D for visualization
4. **Distance metrics**: Measure semantic similarity

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Input
        A[Text Chunks]
    end
    
    subgraph Embedding
        B[SentenceTransformer<br/>all-MiniLM-L6-v2]
        C[384D Vectors]
    end
    
    subgraph Operations
        D[Clustering<br/>K-Means]
        E[Dimensionality Reduction<br/>PCA/UMAP]
        F[Distance Metrics<br/>Cosine Similarity]
    end
    
    subgraph Output
        G[Cluster Assignments]
        H[Low-dim Projections]
        I[Similarity Scores]
    end
    
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    D --> G
    E --> H
    F --> I
    
    style Input fill:#e3f2fd
    style Embedding fill:#fff3e0
    style Operations fill:#e8f5e9
    style Output fill:#f3e5f5
```

---

## 🧮 Core Operations

### 1. Embedding Generation

```python
class TopologyEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize with SentenceTransformer model.
        
        Model produces 384D embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
    
    def encode(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text strings
        
        Returns:
            np.ndarray: shape (len(chunks), 384)
        """
        if not chunks:
            return np.array([]).reshape(0, 384)
        
        embeddings = self.model.encode(
            chunks,
            device=self.device,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Ensure 384D
        if embeddings.shape[1] != 384:
            # Pad or truncate if needed
            embeddings = self._ensure_384d(embeddings)
        
        return embeddings
```

> [!TIP]
> **Robust Fallback**: If `sentence-transformers` is unavailable or fails, the engine automatically falls back to generating **consistent deterministic embeddings** based on content hash. This ensures the system remains functional (e.g., for testing or in restricted environments) even without the heavy model loaded.


**Performance**:
- **Batch encoding**: ~500 chunks/second
- **Single chunk**: ~2ms
- **GPU acceleration**: 3-5x faster

### 2. Clustering

```python
def cluster_concepts(self, embeddings: np.ndarray, n_clusters: int = 10):
    """
    Cluster embeddings using K-Means.
    
    Args:
        embeddings: [n_samples, 384]
        n_clusters: Number of clusters
    
    Returns:
        labels: Cluster assignment for each sample
        centroids: Cluster centers
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    return labels, centroids
```

**Complexity**: O(n × k × i × d)
- n = samples
- k = clusters
- i = iterations
- d = dimensions (384)

### 3. Dimensionality Reduction

```python
def reduce_dimensions(self, embeddings: np.ndarray, method="pca", n_components=2):
    """
    Reduce 384D → 2D/3D for visualization.
    
    Methods:
    - PCA: Fast, linear
    - UMAP: Slower, non-linear, preserves local structure
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=n_components)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced
```

**Use Cases**:
- 2D: Scatter plots, dashboards
- 3D: Interactive network visualizations

### 4. Similarity Metrics

```python
def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between vectors.
    
    Returns: float in [-1, 1]
    - 1.0 = identical
    - 0.0 = orthogonal
    - -1.0 = opposite
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    return dot_product / (norm_a * norm_b + 1e-8)

def batch_similarity(self, query: np.ndarray, corpus: np.ndarray):
    """
    Compute similarity of query against entire corpus.
    
    Optimized with matrix operations.
    """
    # Normalize
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    
    # Dot product
    similarities = np.dot(corpus_norm, query_norm)
    
    return similarities
```

---

## 🔗 Inter-Module Communication

### **Depends On**:

```mermaid
graph LR
    Top[Topology Engine] --> ST[SentenceTransformers]
    Top --> SKL[Scikit-learn]
    Top --> NumPy[NumPy]
    
    style Top fill:#2196F3,color:#fff
    style ST fill:#4CAF50,color:#000
    style SKL fill:#FF9800,color:#000
```

### **Used By**:

```mermaid
graph LR
    SFS[Semantic Memory] --> Top[Topology Engine]
    Causal[Causal Reasoning] --> Top
    Abd[Abduction Engine] --> Top
    UI[Streamlit UI] --> Top
    
    style Top fill:#2196F3,color:#fff
    style SFS fill:#4CAF50,color:#fff
    style Causal fill:#9C27B0,color:#fff
    style Abd fill:#E91E63,color:#fff
```

**1. Semantic Memory**
- **Purpose**: Generate embeddings for chunks
- **Call**: `topology.encode(chunks)`
- **When**: Every document indexing

**2. Causal Reasoning**
- **Purpose**: Cluster concepts
- **Call**: `topology.cluster_concepts(embeddings)`

**3. Abduction Engine**
- **Purpose**: Similarity for gap detection
- **Call**: `topology.cosine_similarity(a, b)`

**4. Streamlit UI**
- **Purpose**: Visualizations
- **Call**: `topology.reduce_dimensions(embeddings, method="umap")`

---

## 🎯 Use Cases

### 1. Embed Text

```python
from core.topology.topology_engine import TopologyEngine

engine = TopologyEngine()

# Single chunk
embedding = engine.encode(["machine learning is awesome"])[0]
print(embedding.shape)  # (384,)

# Batch
texts = ["AI", "ML", "Deep Learning"]
embeddings = engine.encode(texts)
print(embeddings.shape)  # (3, 384)
```

### 2. Find Similar Concepts

```python
# Query
query_emb = engine.encode(["neural networks"])

# Corpus
corpus_texts = ["deep learning", "statistics", "biology"]
corpus_emb = engine.encode(corpus_texts)

# Similarities
sims = engine.batch_similarity(query_emb[0], corpus_emb)
print(sims)  # [0.87, 0.23, 0.15] → "deep learning" most similar
```

### 3. Cluster Documents

```python
# Get embeddings
doc_embeddings = engine.encode(all_documents)

# Cluster
labels, centroids = engine.cluster_concepts(doc_embeddings, n_clusters=10)

# Analyze clusters
for i in range(10):
    docs_in_cluster = [doc for doc, label in zip(all_documents, labels) if label == i]
    print(f"Cluster {i}: {len(docs_in_cluster)} documents")
```

### 4. Visualize Semantic Space

```python
# Reduce to 2D
coords_2d = engine.reduce_dimensions(embeddings, method="umap", n_components=2)

# Plot
import matplotlib.pyplot as plt
plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
plt.show()
```

---

## 📊 Model Details

### SentenceTransformer: all-MiniLM-L6-v2

```python
Model: sentence-transformers/all-MiniLM-L6-v2
Architecture: BERT-based
Parameters: 22.7M
Embedding dimension: 384
Max sequence length: 256 tokens
Training data: 1B+ sentence pairs
```

**Why this model?**
- ✅ Fast: 500 chunks/sec on CPU
- ✅ Compact: 384D vs 768D (BERT-base)
- ✅ High quality: SOTA on semantic textual similarity
- ✅ General purpose: Works across domains

---

## 📈 Performance Benchmarks

| Operation | CPU (i7) | GPU (RTX 3080) |
|-----------|----------|----------------|
| Encode 1 chunk | 2ms | <1ms |
| Encode 100 chunks | ~200ms | ~30ms |
| Encode 1000 chunks | ~2s | ~200ms |
| Cosine similarity (single) | <0.1ms | <0.1ms |
| Batch similarity (1 vs 10K) | ~5ms | ~1ms |
| K-Means (10K points, k=10) | ~500ms | N/A |
| PCA (10K → 2D) | ~100ms | N/A |
| UMAP (10K → 2D) | ~5s | N/A |

---

## ⚙️ Configuration

```python
# Model selection
MODEL_NAME = "all-MiniLM-L6-v2"  # Default
# Alternatives:
# - "paraphrase-multilingual-mpnet-base-v2" (multilingual)
# - "all-mpnet-base-v2" (higher quality, slower)

# Clustering
DEFAULT_N_CLUSTERS = 10
CLUSTERING_ALGO = "kmeans"  # or "dbscan", "hierarchical"

# Dimensionality reduction
DIM_REDUCTION_METHOD = "pca"  # or "umap", "tsne"
```

---

## 🔮 Future Enhancements

- [ ] Multi-lingual support
- [ ] Custom fine-tuning on domain data
- [ ] Hierarchical clustering
- [ ] Online/incremental clustering
- [ ] GPU-accelerated similarity search

---

**Last Updated**: 2025-12-01  
**Version**: 1.0  
**Model**: all-MiniLM-L6-v2  
**Status**: Production


---


# 📄 FILE: docs/core/utils/README.md
================================================================================

# 📦 Utils - Documentação

**Utilitários do sistema Alexandria**

---

## 🌾 Harvester

**Module**: `core/utils/harvester.py` (126 LOC)

Colheitadeira automática de papers científicos via Arxiv.

```python
from core.utils.harvester import ArxivHarvester

harvester = ArxivHarvester()

# Buscar papers
papers = harvester.search_papers("neural compression", max_results=10)

# Harvest completo: busca → download → ingestão
harvester.harvest(
    queries=["VQ-VAE", "predictive coding"],
    max_per_query=5,
    ingest=True  # Ingere no LanceDB
)
```

---

## 🤖 Local LLM

**Module**: `core/utils/local_llm.py` (203 LOC)

Expert Tático com TinyLlama-1.1B para RAG local (zero API cost).

```python
from core.utils.local_llm import LocalLLM

llm = LocalLLM()  # TinyLlama-1.1B-Chat

# Sintetizar resposta a partir de evidências
response = llm.synthesize_facts(
    query="Como funciona VQ-VAE?",
    evidence=retrieved_chunks,
    max_length=512
)
```

**Otimizações**:
- Float32 em CPU (mais rápido que FP16 emulado)
- Multi-threading (8 threads)
- Fallback para concatenação se modelo falhar

---

## 📝 Logger

**Module**: `core/utils/logger.py` (45 LOC)

Logger estruturado com Loguru.

```python
from core.utils.logger import logger

logger.info("Mensagem informativa")
logger.debug("Debug detalhado")  # Só em arquivo
logger.error("Erro crítico")
```

**Saídas**:
- Console: colorido, nível INFO+
- `data/logs/system.log`: JSON estruturado, DEBUG+
- `data/logs/system_readable.log`: texto legível, INFO+

---

**Last Updated**: 2025-12-07


---
