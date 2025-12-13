# Testes Pesados - Tracking File
# Última atualização: 2025-12-13 07:05

## Critérios de "Pesado"
- 🔴 **Muito Pesado**: Carrega modelos reais (VQ-VAE, MonolithV13, SentenceTransformer)
- 🟡 **Médio**: Import torch + módulos core sem mock completo
- 🟢 **Leve**: Apenas mocks ou funções simples

---

## 🔴 Muito Pesados (6 identificados)

| Arquivo | Motivo |
|---------|--------|
| `reasoning/test_vqvae.py` | Import direto de `MonolithV13`, `OrthogonalProductQuantizer` |
| `reasoning/test_model_wiki.py` | Import direto de `MonolithWiki` |
| `reasoning/test_neural_learner.py` | Import torch + neural learner |
| `reasoning/test_mycelial_reasoning.py` | torch + MycelialReasoning real |
| `reasoning/test_mycelial_advanced.py` | torch + MycelialReasoning |
| `memory/test_v11_vision_encoder.py` | torch + CLIP model |

## 🟡 Médios (4 identificados)

| Arquivo | Motivo |
|---------|--------|
| `agents/test_action_agent.py` | Import direto de core.agents.action.* |
| `agents/test_bridge_agent.py` | Import direto de core.agents.* |
| `agents/test_critic_agent.py` | Import direto de core.agents.* |
| `agents/test_executor.py` | Import direto de core.agents.* |

## 🟢 Leves (37 arquivos)

Todos os outros testes que usam `with patch()` para mockar imports pesados.

---

## Progresso: 47/47 analisados

## Recomendação

Para acelerar execução:
```bash
# Rodar apenas testes leves
pytest tests/unit/core -v --ignore=tests/unit/core/reasoning/test_vqvae.py --ignore=tests/unit/core/reasoning/test_model_wiki.py

# Ou usar markers (precisa adicionar @pytest.mark.slow)
pytest tests/unit/core -v -m "not slow"
```
