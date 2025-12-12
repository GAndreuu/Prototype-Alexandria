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
