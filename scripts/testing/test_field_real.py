#!/usr/bin/env python3
"""
Teste REAL do Campo Pré-Estrutural
===================================

Usa embeddings reais do LanceDB para testar o campo.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

print("=" * 70)
print("🔬 TESTE REAL: Campo Pré-Estrutural com LanceDB")
print("=" * 70)

# ============================================================
# 1. Carregar LanceDB
# ============================================================
print("\n📚 [1/5] Conectando ao LanceDB...")

try:
    import lancedb
    
    db = lancedb.connect("data/lancedb_store")
    table = db.open_table("semantic_memory")
    
    total_records = len(table)
    print(f"   ✅ LanceDB conectado: {total_records:,} registros")
    
except Exception as e:
    print(f"   ❌ LanceDB FALHOU: {e}")
    sys.exit(1)

# ============================================================
# 2. Buscar embeddings sobre um tópico
# ============================================================
print("\n🔍 [2/5] Buscando embeddings sobre 'meta-learning'...")

try:
    from core.topology.topology_engine import TopologyEngine
    
    # Inicializa engine para gerar embedding de query
    engine = TopologyEngine()
    
    # Gera embedding da query
    query = "meta-learning free energy principle neural networks"
    query_embedding = engine.encode([query])[0]  # encode retorna array
    
    print(f"   Query: '{query}'")
    print(f"   Embedding shape: {query_embedding.shape}")
    
    # Busca similares
    results = table.search(query_embedding).limit(20).to_pandas()
    
    print(f"   ✅ Encontrados {len(results)} documentos similares")
    
    # Extrai embeddings
    embeddings = np.array([np.array(v) for v in results['vector'].values])
    sources = results['source'].values if 'source' in results.columns else [f"doc_{i}" for i in range(len(results))]
    
    print(f"   Embeddings shape: {embeddings.shape}")
    
except Exception as e:
    print(f"   ❌ Busca FALHOU: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 3. Inicializar Campo
# ============================================================
print("\n⚡ [3/5] Inicializando PreStructuralField...")

try:
    from core.field import PreStructuralField, PreStructuralConfig
    
    config = PreStructuralConfig(
        base_dim=384,  # Dimensão real dos embeddings
        max_expansion=32,
        temperature=1.0,
        configuration_steps=10,
        max_geodesic_steps=10
    )
    
    field = PreStructuralField(config)
    
    print(f"   ✅ Campo inicializado: {config.base_dim}D")
    
except Exception as e:
    print(f"   ❌ Campo FALHOU: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 4. Trigger embeddings reais
# ============================================================
print("\n🎯 [4/5] Triggando conceitos no campo...")

try:
    for i, (emb, source) in enumerate(zip(embeddings[:10], sources[:10])):
        # Normaliza
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        
        # Trigger
        state = field.trigger(emb_norm, intensity=0.8)
        
        # Nome do arquivo (truncado)
        name = str(source).split('/')[-1][:40] if source else f"doc_{i}"
        
        print(f"   [{i+1}/10] {name}...")
        print(f"          F_mean={state.mean_free_energy:.4f}, atratores={state.num_attractors}")
    
    print(f"\n   ✅ {len(embeddings[:10])} conceitos triggados")
    
except Exception as e:
    print(f"   ❌ Trigger FALHOU: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 5. Propagar e Cristalizar
# ============================================================
print("\n💎 [5/5] Propagando e cristalizando...")

try:
    # Propaga
    print("   Propagando dinâmica...")
    states = field.propagate(steps=5)
    
    print(f"   → {len(states)} estados gerados")
    
    # Annealing (opcional - exploration → exploitation)
    print("   Aplicando annealing...")
    annealed_states = field.anneal(start_temp=2.0, end_temp=0.5, steps=10)
    
    print(f"   → {len(annealed_states)} estados após annealing")
    
    # Cristaliza
    print("   Cristalizando...")
    graph = field.crystallize()
    
    print(f"\n   ✅ Cristalização concluída")
    print(f"      Nós (atratores): {len(graph['nodes'])}")
    print(f"      Arestas: {len(graph['edges'])}")
    
    # Mostra atratores
    if graph['nodes']:
        print("\n   Atratores encontrados:")
        for node in graph['nodes'][:5]:
            print(f"      - {node['id']}: F={node['free_energy']:.4f}")
    
except Exception as e:
    print(f"   ❌ Propagação/Cristalização FALHOU: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Estatísticas Finais
# ============================================================
print("\n" + "=" * 70)
print("📊 ESTATÍSTICAS FINAIS")
print("=" * 70)

try:
    stats = field.stats()
    print(f"""
   CAMPO PRÉ-ESTRUTURAL
   ====================
   
   Manifold:
     - Dimensão: {stats['manifold']['current_dim']}
     - Pontos: {stats['manifold']['num_points']}
     - Pontos ativos: {stats['manifold']['active_points']}
   
   Metric:
     - Deformações: {stats['metric']['deformations']}
   
   Field:
     - Temperatura final: {stats['field']['temperature']:.2f}
     - F médio: {stats['field'].get('mean_F', 'N/A')}
     - Atratores: {stats['field']['num_attractors']}
   
   Operações:
     - Triggers: {stats['triggers']}
     - Ciclos: {stats['cycles_completed']}
   
   INTERPRETAÇÃO:
   - Cada trigger deforma o campo criando um "poço" de atração
   - F baixo = conceito bem ancorado
   - Atratores = clusters emergentes
   - Após cristalização, essas estruturas podem alimentar o Mycelial
""")
except Exception as e:
    print(f"   Stats FALHOU: {e}")

print("=" * 70)
print("✅ TESTE REAL CONCLUÍDO")
print("=" * 70)
