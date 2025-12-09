#!/usr/bin/env python3
"""
Teste REAL SIMPLES do Campo Pré-Estrutural
==========================================
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
sys.path.insert(0, '.')

print("=" * 60)
print("TESTE REAL: Campo Pre-Estrutural com LanceDB")
print("=" * 60)

# 1. LanceDB
print("\n[1] Conectando LanceDB...")
import lancedb
db = lancedb.connect("data/lancedb_store")
table = db.open_table("semantic_memory")
print(f"    OK: {len(table):,} registros")

# 2. Buscar embeddings
print("\n[2] Buscando embeddings relevantes...")
from core.topology.topology_engine import TopologyEngine
engine = TopologyEngine()

query = "meta-learning free energy"
query_emb = engine.encode([query])[0]
results = table.search(query_emb).limit(10).to_pandas()
embeddings = np.array([np.array(v) for v in results['vector'].values])
print(f"    OK: {embeddings.shape}")

# 3. Campo
print("\n[3] Inicializando Campo...")
from core.field import PreStructuralField, PreStructuralConfig

config = PreStructuralConfig(base_dim=384)
field = PreStructuralField(config)
print(f"    OK: {config.base_dim}D")

# 4. Triggers
print("\n[4] Triggando conceitos...")
for i, emb in enumerate(embeddings[:5]):
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    state = field.trigger(emb_norm, intensity=0.8)
    print(f"    [{i+1}] F_mean={state.mean_free_energy:.4f}")

# 5. Cristalizar
print("\n[5] Cristalizando...")
graph = field.crystallize()
print(f"    Nos: {len(graph['nodes'])}")
print(f"    Arestas: {len(graph['edges'])}")

# 6. Stats
print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
stats = field.stats()
print(f"""
    Manifold: {stats['manifold']['current_dim']}D, {stats['manifold']['num_points']} pontos
    Metric: {stats['metric']['deformations']} deformacoes
    Field: T={stats['field']['temperature']}, {stats['field']['num_attractors']} atratores
    Triggers: {stats['triggers']}
""")
print("SUCESSO!")
