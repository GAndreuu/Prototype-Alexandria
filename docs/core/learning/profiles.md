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
