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
