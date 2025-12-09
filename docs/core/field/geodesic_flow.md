# GeodesicFlow (`core/field/geodesic_flow.py`)

> Propagação via caminhos geodésicos.

## Visão Geral

O `GeodesicFlow` computa geodésicas — os caminhos mais curtos em espaço curvo. São análogos a "linhas retas" em espaço plano, mas seguem a curvatura.

## Equação Geodésica

```
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
```

Onde `Γ` são os símbolos de Christoffel da métrica.

## Uso

```python
from core.field import GeodesicFlow, GeodesicConfig

config = GeodesicConfig(
    max_steps=100,
    step_size=0.01,
    use_scipy_integrator=True  # Mais preciso, mais lento
)
flow = GeodesicFlow(manifold, metric, config)

# Geodésica a partir de ponto e velocidade
start = point.coordinates
velocity = np.random.randn(dim) * 0.1
path = flow.compute_geodesic(start, velocity, max_steps=50)

print(f"Passos: {path.n_steps}")
print(f"Comprimento: {path.length}")
print(f"Convergiu: {path.converged}")

# Caminho mais curto entre dois pontos
shortest = flow.shortest_path(start, end)
```

## GeodesicPath

```python
@dataclass
class GeodesicPath:
    points: np.ndarray      # [n_steps, dim]
    velocities: np.ndarray  # [n_steps, dim]
    length: float
    n_steps: int
    converged: bool
```

## Casos de Uso

1. **Propagação de ativação**: Conceito ativado propaga ao longo de geodésicas
2. **Conexão de clusters**: Encontrar caminho natural entre conceitos distantes
3. **Visualização**: Mostrar "linhas de fluxo" no espaço de conhecimento
