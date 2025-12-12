# 🧪 Validação Científica: Baseline vs Alexandria

Este script compara a performance do sistema Alexandria contra um Baseline simples (KMeans) na tarefa de encontrar conexões conceituais conhecidas (Ground Truth).

## Uso

```bash
python scripts/validate_alexandria.py
```

## Metodologia

### Ground Truth
Baseado em conexões estabelecidas na literatura de Active Inference e Free Energy Principle (FEP).
- Ex: "Active Inference" <-> "Control Theory"
- Ex: "Predictive Coding" <-> "Attention"

### Baseline
- **Algoritmo**: KMeans (10 clusters)
- **Critério**: Proximidade Euclidiana entre papers de clusters diferentes.

### Alexandria
- **Field**: PreStructuralField (Geometria Diferencial)
- **VQ-VAE**: Códigos discretos compartilhados
- **Mycelial**: Co-ativação Hebbiana

## Métricas
- **Recall**: % de conexões do Ground Truth encontradas.
- **Unique Connections**: Novas conexões encontradas por cada método.
