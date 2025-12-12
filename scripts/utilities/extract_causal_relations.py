"""
Extração de Relações Causais dos Papers (OTIMIZADO)
=====================================================

Versão paralela usando multiprocessing para máximo desempenho.

Uso:
    python scripts/utilities/extract_causal_relations.py
"""

import sys
import os
import re
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


# Padrões pré-compilados (mais rápido)
CAUSAL_PATTERNS = [
    re.compile(r"(\w+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:induces?|triggers?|enables?)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:prevents?|inhibits?|promotes?)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:increases?|decreases?|affects?)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:influences?|modulates?|requires?)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+depends\s+on\s+(\w+)", re.I),
]

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'it', 'they', 'we', 'you', 'which', 'what', 'who', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'some', 'such', 'also',
    'now', 'here', 'there', 'then', 'if', 'or', 'and', 'but', 'as',
    'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'one', 'two', 'first', 'new', 'high', 'low', 'well', 'more', 'most',
    'other', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'data',
    'using', 'based', 'paper', 'study', 'method', 'approach', 'model', 'results'
})


def process_chunk(content: str) -> list:
    """Processa um chunk e extrai relações (worker function)"""
    if not content or len(content) < 50:
        return []
    
    relations = []
    
    for pattern in CAUSAL_PATTERNS:
        for match in pattern.finditer(content):
            cause = match.group(1).lower()
            effect = match.group(2).lower()
            
            if (cause not in STOPWORDS and effect not in STOPWORDS 
                and 3 < len(cause) < 20 and 3 < len(effect) < 20
                and cause.isalpha() and effect.isalpha()):
                relations.append((cause, effect))
    
    return relations


def main():
    print("=" * 60)
    print("🔗 EXTRAÇÃO DE RELAÇÕES CAUSAIS (PARALELO)")
    print("=" * 60)
    
    n_workers = cpu_count()
    print(f"   Usando {n_workers} workers\n")
    
    # Carregar dados
    print("📦 Carregando dados do LanceDB...")
    import lancedb
    
    db_path = os.path.join(project_root, "data", "lancedb_store")
    db = lancedb.connect(db_path)
    table = db.open_table("semantic_memory")
    
    df = table.to_pandas()
    contents = df['content'].dropna().tolist()
    total = len(contents)
    print(f"   ✅ {total} chunks carregados\n")
    
    # Processar em paralelo
    print("🔍 Extraindo relações em paralelo...")
    
    causal_counts = defaultdict(lambda: defaultdict(int))
    
    with Pool(n_workers) as pool:
        # Processar em chunks para mostrar progresso
        chunk_size = 10000
        processed = 0
        
        for i in range(0, total, chunk_size):
            batch = contents[i:i+chunk_size]
            results = pool.map(process_chunk, batch)
            
            # Agregar resultados
            for relations in results:
                for cause, effect in relations:
                    causal_counts[cause][effect] += 1
            
            processed += len(batch)
            pct = (processed / total) * 100
            print(f"   Progresso: {processed}/{total} ({pct:.0f}%)")
    
    print(f"\n📊 Conceitos únicos: {len(causal_counts)}")
    
    # Filtrar relações fracas
    print("🔧 Filtrando (min 2 ocorrências)...")
    
    filtered_graph = {}
    total_edges = 0
    
    for cause, effects in causal_counts.items():
        strong = {e: c for e, c in effects.items() if c >= 2}
        if strong:
            filtered_graph[cause] = strong
            total_edges += len(strong)
    
    print(f"   Nós: {len(filtered_graph)}")
    print(f"   Arestas: {total_edges}")
    
    # Salvar
    causal_path = os.path.join(project_root, "data", "causal_graph.json")
    
    print(f"\n💾 Salvando...")
    with open(causal_path, 'w') as f:
        json.dump(filtered_graph, f, indent=2)
    
    size_kb = os.path.getsize(causal_path) / 1024
    print(f"   ✅ Salvo ({size_kb:.1f} KB)")
    
    # Top relações
    print("\n📋 Top relações:")
    all_relations = []
    for cause, effects in filtered_graph.items():
        for effect, count in effects.items():
            all_relations.append((cause, effect, count))
    
    all_relations.sort(key=lambda x: x[2], reverse=True)
    for cause, effect, count in all_relations[:10]:
        print(f"   {cause} → {effect} ({count}x)")
    
    print("\n" + "=" * 60)
    print("✅ CONCLUÍDO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
