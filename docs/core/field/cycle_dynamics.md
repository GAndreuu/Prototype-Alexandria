# CycleDynamics (`core/field/cycle_dynamics.py`)

> Ciclo Expansão → Configuração → Compressão.

## Visão Geral

O `CycleDynamics` é o coração do Campo Pré-Estrutural. Implementa um ciclo de três fases que permite ao sistema descobrir estrutura:

```
      ┌─────────────────┐
      │    EXPANSÃO     │  ← Espaço cresce
      │  (mais dims)    │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │  CONFIGURAÇÃO   │  ← Elementos se arranjam
      │  (annealing)    │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │   COMPRESSÃO    │  ← Dimensões colapsam
      │  (menos dims)   │
      └────────┬────────┘
               │
               ▼
        estrutura emerge
```

## Fases

### 1. EXPANSÃO

- Dimensões são adicionadas
- Graus de liberdade aumentam
- Permite novas configurações

### 2. CONFIGURAÇÃO

- Simulated annealing
- Temperatura decresce gradualmente
- Elementos encontram posições de mínima energia

### 3. COMPRESSÃO

- Dimensões com baixa variância são removidas
- Informação é comprimida
- Estrutura densa emerge

## Uso

```python
from core.field import CycleDynamics, CycleConfig

config = CycleConfig(
    configuration_steps=50,
    expansion_threshold=0.7,    # F > 0.7 → expande
    compression_threshold=0.3,  # F < 0.3 → comprime
    max_expansion_dims=32
)

cycle = CycleDynamics(manifold, metric, field, flow, config)

# Rodar ciclo
trigger = np.random.randn(384)  # Embedding que inicia
result = cycle.run_cycle(trigger)

print(f"Fase: {result.phase}")
print(f"Ciclo #: {result.cycle_number}")
print(f"Dims adicionadas: {result.dimensions_added}")
print(f"Dims removidas: {result.dimensions_removed}")
print(f"ΔF: {result.free_energy_delta}")
```

## CycleState

```python
@dataclass
class CycleState:
    cycle_number: int
    phase: CyclePhase
    dimensions_added: int
    dimensions_removed: int
    free_energy_delta: float
    configuration_loss: float
    timestamp: str
```

## Cristalização

Após um ciclo, a estrutura pode ser "cristalizada" em grafo:

```python
graph = cycle.crystallize()
# {"nodes": [...], "edges": [...]}
```

## Auto-modificação

O ciclo pode modificar suas próprias regras:

```python
# Após N ciclos, regras de transição são atualizadas
cycle.update_rules(new_expansion_threshold=0.6)
```
