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
