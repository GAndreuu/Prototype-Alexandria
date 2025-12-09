# RiemannianMetric (`core/field/metric.py`)

> Métrica que deforma localmente com ativação.

## Visão Geral

A `RiemannianMetric` mede distâncias em um espaço curvo. Quando conceitos são ativados, eles **deformam** o espaço ao redor, criando "poços" de atração.

## Matemática

### Métrica Deformada

```
g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)
```

Onde:
- `δ_ij` = métrica plana (identidade)
- `w_a` = intensidade da deformação a
- `c_a` = centro da deformação a
- `r` = raio de influência

### Curvatura

A curvatura escalar R indica "quão curvado" está o espaço:
- R = 0: espaço plano
- R > 0: espaço curvado positivamente (esfera)
- R < 0: espaço curvado negativamente (sela)

## Uso

```python
from core.field import RiemannianMetric, MetricConfig, DynamicManifold

# Criar
manifold = DynamicManifold()
config = MetricConfig(
    deformation_radius=0.3,
    deformation_strength=0.5
)
metric = RiemannianMetric(manifold, config)

# Deformar em um ponto
point = manifold.points["concept_1"]
metric.deform_at(point.coordinates, intensity=0.8)

# Medir distância (não-Euclidiana)
dist = metric.distance(point_a, point_b)

# Curvatura local
R = metric.curvature_scalar_at(point.coordinates)

# Relaxar deformações
metric.relax(rate=0.1)  # Decai 10%

# Stats
print(metric.stats())
# {"deformations": 5, "total_curvature": 0.23}
```

## Símbolos de Christoffel

Para geodésicas, calculamos os símbolos de Christoffel:

```python
Γ = metric.christoffel_at(point.coordinates)
# Tensor [dim, dim, dim]
```

Esses símbolos definem como vetores mudam ao se mover no espaço curvo.

## Integração

A métrica é usada pelo:
- `GeodesicFlow`: para propagar no espaço curvo
- `FreeEnergyField`: para calcular gradientes
- `CycleDynamics`: para detectar atratores
