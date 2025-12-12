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
