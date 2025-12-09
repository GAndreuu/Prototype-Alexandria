# Campo Pré-Estrutural (`core/field/`)

> **Camada geométrica que opera no nível de potencial, não de estrutura.**

---

## Visão Geral

O Campo Pré-Estrutural é uma camada de processamento que trata o conhecimento como um **campo contínuo** em vez de estruturas discretas (grafos, tabelas). Ele unifica Learning e Reasoning através de geometria diferencial.

### Metáfora

| Sistema Tradicional | Campo Pré-Estrutural |
|---------------------|---------------------|
| Mapa de ruas | Campo gravitacional |
| Conexões fixas | Potencial que muda |
| "Onde as coisas estão" | "Para onde querem ir" |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    PreStructuralField                        │
│         (wrapper unificado com conexões VQ-VAE/Mycelial)    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │   Manifold  │───▶│   Metric    │───▶│    Field     │   │
│   │  (espaço)   │    │ (distâncias)│    │   F(x,t)     │   │
│   └─────────────┘    └─────────────┘    └──────────────┘   │
│          │                   │                  │           │
│          ▼                   ▼                  ▼           │
│   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │  GeodesicFlow│   │CycleDynamics│───▶│ Cristalização│   │
│   │ (propagação)│    │(Exp→Cfg→Cmp)│    │   (→grafo)   │   │
│   └─────────────┘    └─────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes

### 1. DynamicManifold (`manifold.py`)

Variedade diferenciável com dimensão variável.

```python
from core.field import DynamicManifold

manifold = DynamicManifold()
point = manifold.embed(embedding_384d)
manifold.expand_dimension(n_dims=4)  # Cresce durante aprendizado
manifold.contract_dimension(n_dims=2)  # Comprime após consolidar
```

**Responsabilidades:**
- Projetar embeddings 384D em pontos da variedade
- Gerenciar dimensões dinâmicas (expansão/contração)
- Manter estrutura de vizinhança (KDTree)

---

### 2. RiemannianMetric (`metric.py`)

Métrica que deforma localmente quando conceitos são ativados.

```python
from core.field import RiemannianMetric

metric = RiemannianMetric(manifold)
metric.deform_at(point.coordinates, intensity=0.8)
distance = metric.distance(point_a, point_b)
curvature = metric.curvature_scalar_at(point)
```

**Responsabilidades:**
- Medir distâncias em espaço curvo
- Deformar espaço quando conceitos são "triggerados"
- Calcular curvatura (Ricci scalar)
- Computar símbolos de Christoffel para geodésicas

---

### 3. FreeEnergyField (`free_energy_field.py`)

Campo de energia livre F(x) = E(x) - T·S(x).

```python
from core.field import FreeEnergyField

field = FreeEnergyField(manifold, metric)
F = field.free_energy_at(point)
gradient = field.gradient_at(point)
attractors = field.get_state().attractors
```

**Fórmula:**
- **E(x)**: Energia interna (surpresa/prediction error)
- **S(x)**: Entropia (incerteza sobre transições)
- **T**: Temperatura (exploration vs exploitation)

**Responsabilidades:**
- Calcular energia livre em cada ponto
- Encontrar atratores (mínimos locais)
- Computar gradientes para dinâmica

---

### 4. GeodesicFlow (`geodesic_flow.py`)

Propagação via caminhos geodésicos (mais curtos em espaço curvo).

```python
from core.field import GeodesicFlow

flow = GeodesicFlow(manifold, metric)
path = flow.compute_geodesic(start, velocity)
shortest = flow.shortest_path(start, end)
```

**Responsabilidades:**
- Computar geodésicas (equação: d²x/dt² = -Γ v v)
- Propagar ativação seguindo geometria
- Gerar streamlines para visualização

---

### 5. CycleDynamics (`cycle_dynamics.py`)

O coração do sistema: ciclo Expansão → Configuração → Compressão.

```python
from core.field import CycleDynamics

cycle = CycleDynamics(manifold, metric, field, flow)
result = cycle.run_cycle(trigger_embedding)
```

**Fases:**
1. **Expansão**: Espaço cresce em dimensões novas
2. **Configuração**: Elementos se arranjam (annealing)
3. **Compressão**: Colapsa em estrutura densa

**Responsabilidades:**
- Orquestrar ciclo completo
- Atualizar regras de transição (meta-learning)
- Cristalizar campo em grafo

---

### 6. PreStructuralField (`pre_structural_field.py`)

Wrapper unificado que conecta com VQ-VAE e Mycelial.

```python
from core.field import PreStructuralField

# Inicializa
field = PreStructuralField()

# Conecta com resto do sistema
field.connect_vqvae(vqvae_model)
field.connect_mycelial(mycelial_reasoning)

# Usa
state = field.trigger(embedding)
states = field.propagate(steps=5)
graph = field.crystallize()

# Stats
print(field.stats())
```

---

## Uso Típico

```python
from core.field import PreStructuralField, PreStructuralConfig

# 1. Configurar
config = PreStructuralConfig(
    base_dim=384,
    temperature=1.0,
    max_expansion=32
)
field = PreStructuralField(config)

# 2. Trigger conceitos
for embedding in embeddings:
    state = field.trigger(embedding, intensity=0.8)
    print(f"F_mean = {state.mean_free_energy:.4f}")

# 3. Propagar dinâmica
states = field.propagate(steps=10)

# 4. Annealing (exploration → exploitation)
final_states = field.anneal(start_temp=2.0, end_temp=0.1, steps=50)

# 5. Cristalizar em grafo
graph = field.crystallize()
print(f"Atratores: {len(graph['nodes'])}")
print(f"Conexões: {len(graph['edges'])}")
```

---

## Integração com Alexandria

### Conexão com VQ-VAE
```python
field.connect_vqvae(vqvae_model)
# → Codebook vira anchor_points na variedade
```

### Conexão com Mycelial
```python
field.connect_mycelial(mycelial)
# → Cristalização alimenta grafo Hebbiano
```

### Conexão com VariationalFreeEnergy
```python
field.connect_variational_fe(vfe_module)
# → Sincroniza beliefs e componentes de F
```

---

## Arquivos

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `__init__.py` | 54 | Exports do módulo |
| `manifold.py` | 400 | DynamicManifold |
| `metric.py` | 436 | RiemannianMetric |
| `free_energy_field.py` | 500 | FreeEnergyField |
| `geodesic_flow.py` | 551 | GeodesicFlow |
| `cycle_dynamics.py` | 592 | CycleDynamics |
| `pre_structural_field.py` | 400 | PreStructuralField |

**Total: ~3,000 LOC**

---

## Dependências

- numpy
- scipy (KDTree, integrate, ndimage)
- typing, dataclasses, logging

---

## Testes

```bash
python scripts/testing/test_field.py       # Teste rápido
python scripts/testing/test_field_simple.py # Teste com LanceDB real
```

---

## Matemática Base

### Energia Livre
```
F(x) = E(x) - T·S(x)

Onde:
- E = energia interna (surpresa)
- S = entropia
- T = temperatura
```

### Equação Geodésica
```
d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0

Onde Γ = símbolos de Christoffel da métrica
```

### Deformação da Métrica
```
g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)

Onde:
- δ_ij = métrica plana (identidade)
- w_a = intensidade da deformação a
- c_a = centro da deformação a
- r = raio de influência
```
