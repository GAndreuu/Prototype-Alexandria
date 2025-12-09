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
