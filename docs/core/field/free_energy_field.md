# FreeEnergyField (`core/field/free_energy_field.py`)

> Campo F(x) = E(x) - T·S(x) sobre a variedade.

## Visão Geral

O `FreeEnergyField` computa a energia livre em cada ponto da variedade. É o princípio unificador que determina para onde o sistema "quer ir".

## Fórmula

```
F(x) = E(x) - T·S(x)
```

| Componente | Símbolo | Descrição |
|------------|---------|-----------|
| Energia Livre | F | O que minimizamos |
| Energia Interna | E | Surpresa/prediction error |
| Entropia | S | Incerteza sobre transições |
| Temperatura | T | Trade-off exploration/exploitation |

## Classes

### FieldConfig

```python
@dataclass
class FieldConfig:
    temperature: float = 1.0        # T
    energy_scale: float = 1.0       # Escala de E
    entropy_scale: float = 1.0      # Escala de S
    gradient_step: float = 0.01     # Para gradientes numéricos
```

### FieldState

```python
@dataclass
class FieldState:
    timestamp: str
    mean_free_energy: float
    min_free_energy: float
    max_free_energy: float
    num_attractors: int
    attractors: List[np.ndarray]
    gradient_field: np.ndarray
```

## Uso

```python
from core.field import FreeEnergyField, FieldConfig

field = FreeEnergyField(manifold, metric)

# Energia em um ponto
F = field.free_energy_at(point.coordinates)
E = field.energy_at(point.coordinates)
S = field.entropy_at(point.coordinates)

# Gradiente (direção de descida)
grad = field.gradient_at(point.coordinates)

# Estado do campo inteiro
state = field.compute_field()
print(f"F_mean = {state.mean_free_energy}")
print(f"Atratores = {state.num_attractors}")

# Ajustar temperatura
field.set_temperature(0.5)  # Mais exploitation
field.set_temperature(2.0)  # Mais exploration

# Stats
stats = field.stats()
```

## Temperatura

| T | Comportamento |
|---|---------------|
| T → 0 | Sistema vai para mínimos de energia (greedy) |
| T = 1 | Balanceado |
| T → ∞ | Sistema ignora energia, maximiza entropia (random) |

## Atratores

Atratores são **mínimos locais** de F:

```python
attractors = field.get_state().attractors
# Lista de coordenadas onde ∇F ≈ 0 e F é mínimo local
```
