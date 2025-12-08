"""
Script para Construir Grafo Causal
===================================

Executa:
1. Treina TopologyEngine com vetores do LanceDB
2. Constrói grafo causal via CausalEngine
3. Descobre variáveis latentes e lacunas

Uso:
    python scripts/utilities/build_causal_graph.py
"""

import sys
import os
import numpy as np

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vectors_from_lancedb():
    """Carrega vetores do LanceDB para treinamento"""
    import lancedb
    
    db_path = os.path.join(project_root, "data", "lancedb_store")
    db = lancedb.connect(db_path)
    
    tables = db.table_names()
    print(f"   Tabelas encontradas: {tables}")
    
    all_vectors = []
    
    for table_name in tables:
        try:
            table = db.open_table(table_name)
            df = table.to_pandas()
            if 'vector' in df.columns:
                vectors = np.array(df['vector'].tolist())
                all_vectors.append(vectors)
                print(f"   {table_name}: {len(vectors)} vetores")
        except Exception as e:
            print(f"   {table_name}: erro - {e}")
    
    if all_vectors:
        return np.vstack(all_vectors)
    return np.array([])


def main():
    print("=" * 60)
    print("🔗 CONSTRUÇÃO DO GRAFO CAUSAL")
    print("=" * 60)
    
    # 1. Carregar TopologyEngine
    print("\n📐 Carregando Topology Engine...")
    from core.topology.topology_engine import TopologyEngine
    topology = TopologyEngine()
    print("   ✅ OK")
    
    # 2. Verificar se já está treinado
    if not topology.is_trained:
        print("\n🔧 TopologyEngine não treinado. Iniciando treinamento...")
        
        # Carregar vetores do LanceDB
        print("\n📦 Carregando vetores do LanceDB...")
        vectors = load_vectors_from_lancedb()
        
        if len(vectors) == 0:
            print("   ❌ Nenhum vetor encontrado!")
            return None
        
        print(f"   ✅ {len(vectors)} vetores carregados")
        
        # Treinar manifold
        print("\n🎓 Treinando Manifold (clustering)...")
        n_clusters = min(256, len(vectors) // 10)  # Max 256 clusters
        result = topology.train_manifold(vectors, n_clusters=n_clusters)
        print(f"   ✅ {result.get('n_clusters', 0)} clusters criados")
        
        # Salvar topologia
        topology.save_topology(os.path.join(project_root, "data", "topology.json"))
        print("   ✅ Topologia salva")
    else:
        print("\n✅ TopologyEngine já treinado")
    
    # 3. Carregar SemanticFileSystem
    print("\n🧠 Carregando Semantic Memory...")
    from core.memory.semantic_memory import SemanticFileSystem
    memory = SemanticFileSystem(topology)
    print("   ✅ OK")
    
    # 4. Criar CausalEngine
    print("\n🔮 Inicializando Causal Engine...")
    from core.reasoning.causal_reasoning import CausalEngine
    causal = CausalEngine(topology, memory)
    print("   ✅ OK")
    
    # 5. Construir Grafo Causal
    print("\n" + "=" * 60)
    print("🔨 Construindo Grafo Causal...")
    print("=" * 60 + "\n")
    
    graph = causal.build_causal_graph()
    
    # 6. Mostrar estatísticas
    print("\n" + "=" * 60)
    print("📊 ESTATÍSTICAS")
    print("=" * 60)
    
    stats = causal.get_statistics()
    print(f"\n📈 Grafo Causal:")
    print(f"   Nós: {stats.get('num_nodes', 0)}")
    print(f"   Arestas: {stats.get('num_edges', 0)}")
    print(f"   Densidade: {stats.get('density', 0):.4f}")
    
    # 7. Descobrir variáveis latentes
    print("\n🔮 Descobrindo Variáveis Latentes...")
    latent = causal.discover_latent_variables()
    print(f"   Variáveis latentes: {len(latent)}")
    
    # 8. Identificar lacunas lógicas
    print("\n🕳️ Identificando Lacunas Lógicas...")
    gaps = causal.identify_logic_gaps()
    print(f"   Lacunas identificadas: {len(gaps)}")
    
    print("\n" + "=" * 60)
    print("✅ GRAFO CAUSAL CONSTRUÍDO!")
    print("=" * 60)
    print(f"\n💾 Salvo em: data/causal_graph.json")
    
    return graph


if __name__ == "__main__":
    main()
