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
