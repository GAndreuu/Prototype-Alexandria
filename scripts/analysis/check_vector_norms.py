import sys
import os
from pathlib import Path
import numpy as np
import lancedb

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

def check_norms():
    print("🔍 Connecting to LanceDB...")
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    db = lancedb.connect(db_path)
    table_name = "semantic_memory"
    
    if table_name not in db.table_names():
        print(f"❌ Table '{table_name}' not found.")
        return

    table = db.open_table(table_name)
    
    print("📦 Fetching sample vectors...")
    # Fetch first 1000 vectors
    df = table.search(np.random.rand(384)).limit(1000).to_pandas()
    
    vectors = np.stack(df['vector'].values)
    norms = np.linalg.norm(vectors, axis=1)
    
    print("\n📊 Vector Norm Analysis:")
    print("=" * 40)
    print(f"Sample Size: {len(norms)}")
    print(f"Mean Norm:   {np.mean(norms):.4f}")
    print(f"Min Norm:    {np.min(norms):.4f}")
    print(f"Max Norm:    {np.max(norms):.4f}")
    print(f"Std Dev:     {np.std(norms):.4f}")
    print("-" * 40)
    
    # Debug specific query
    print("\n🕵️ Debugging Specific Query:")
    query_text = "vector quantization power law neural criticality"
    
    # We need the encoder to generate the query vector
    from core.topology.topology_engine import TopologyEngine
    encoder = TopologyEngine()
    query_vec = encoder.encode([query_text])[0]
    
    # Normalize query
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    print(f"🔍 Query Vector Sample (first 5): {query_vec[:5]}")
    with open("vector_debug_check.txt", "w") as f:
        f.write(str(query_vec.tolist()))
    
    results = table.search(query_vec).limit(5).to_list()
    
    print(f"Query: '{query_text}'")
    for i, r in enumerate(results):
        dist = r.get('_distance', -1)
        relevance = max(0, 1 - (dist / 2))
        print(f"[{i+1}] Dist: {dist:.6f} | Calc Relevance: {relevance:.4f} | Source: {r['source']}")

if __name__ == "__main__":
    check_norms()
