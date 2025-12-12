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
