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
