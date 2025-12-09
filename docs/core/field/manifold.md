# DynamicManifold (`core/field/manifold.py`)

> Variedade diferenciável com dimensão variável.

## Visão Geral

O `DynamicManifold` representa o espaço onde o conhecimento vive. Não é um grafo discreto — é um **espaço contínuo** que pode expandir e contrair suas dimensões.

## Conceitos

| Conceito | Descrição |
|----------|-----------|
| Pontos | Conceitos/embeddings projetados na variedade |
| Códigos VQ-VAE | Coordenadas discretas (âncoras) |
| Dimensão | Pode crescer (expansão) e encolher (compressão) |
| Topologia | Emerge da distribuição de pontos |

## Classes

### ManifoldConfig

```python
@dataclass
class ManifoldConfig:
    base_dim: int = 384           # Dimensão base (embedding)
    num_heads: int = 4            # Heads do VQ-VAE
    codebook_size: int = 256      # Códigos por head
    max_expansion: int = 128      # Máximo de dimensões extras
    sparsity_threshold: float = 0.01
    neighborhood_k: int = 16
```

### ManifoldPoint

```python
@dataclass
class ManifoldPoint:
    coordinates: np.ndarray      # Coordenadas contínuas [dim]
    discrete_codes: np.ndarray   # Códigos VQ-VAE [4]
    activation: float = 0.0      # Nível de ativação
    metadata: Dict[str, Any]
```

## Uso

```python
from core.field import DynamicManifold, ManifoldConfig

# Criar variedade
config = ManifoldConfig(base_dim=384)
manifold = DynamicManifold(config)

# Projetar embedding
embedding = np.random.randn(384)
point = manifold.embed(embedding)

# Adicionar à variedade
manifold.add_point("concept_1", point)

# Ativar (trigger)
manifold.activate_point("concept_1", intensity=0.8)

# Buscar vizinhos
neighbors = manifold.get_neighbors(point, k=5)
# [(point_id, distance), ...]

# Expandir dimensões
manifold.expand_dimension(n_dims=4)
print(manifold.current_dim)  # 388

# Contrair
manifold.contract_dimension(n_dims=2)
print(manifold.current_dim)  # 386

# Estatísticas
stats = manifold.stats()
# {
#     "num_points": 100,
#     "current_dim": 386,
#     "active_points": 15,
#     "mean_activation": 0.23
# }
```

## Integração com VQ-VAE

```python
# Definir pontos âncora do codebook
manifold.set_anchor_points(vqvae.get_codebook())

# Criar ponto a partir de códigos
point = manifold.from_vqvae_codes([12, 45, 200, 78])
```

## Serialização

```python
# Salvar
data = manifold.to_dict()

# Carregar
manifold = DynamicManifold.from_dict(data)
```
