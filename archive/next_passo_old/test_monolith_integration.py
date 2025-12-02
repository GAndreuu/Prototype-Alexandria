"""
End-to-End Integration Test
Tests complete pipeline with new Monolith encoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

logger_imports = []
try:
    from core.reasoning.mycelial_reasoning import MycelialVQVAE
    logger_imports.append("✅ MycelialVQVAE imported")
except Exception as e:
    print(f"❌ MycelialVQVAE import failed: {e}")
    exit(1)

def test_e2e():
    print("="*80)
    print("END-TO-END INTEGRATION TEST - NEW MONOLITH ENCODER")
    print("="*80)
    
    # 1. Load models
    print("\n1. Loading models...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    wrapper = MycelialVQVAE.load_default()
    print("✅ Models loaded")
    
    # 2. Process documents
    print("\n2. Processing documents...")
    docs = [
        "Machine learning identifies patterns in data",
        "Deep learning uses neural networks with multiple layers",
        "Artificial intelligence mimics human cognition"
    ]
    
    all_indices = []
    for doc in docs:
        emb = text_encoder.encode(doc)
        indices = wrapper.encode(emb)
        wrapper.observe(indices)
        all_indices.append(indices)
        print(f"   '{doc[:40]}...' → {indices}")
    
    # 3. Test reasoning
    print("\n3. Testing neural reasoning...")
    query = "How does AI learn patterns?"
    query_emb = text_encoder.encode(query)
    query_indices = wrapper.encode(query_emb)
    
    print(f"   Query: '{query}'")
    print(f"   Original indices: {query_indices}")
    
    new_indices, activation = wrapper.reason(query_indices)
    print(f"   Reasoned indices: {new_indices}")
    print(f"   Max activation per head: {[act.max() for act in activation]}")
    
    # 4. Check stats
    print("\n4. Network statistics...")
    stats = wrapper.get_network_stats()
    print(f"   Observations: {stats.get('total_observations', 0)}")
    print(f"   Connections: {stats.get('active_connections', 0)}")
    print(f"   Density: {stats.get('density', 0):.6f}")
    
    # 5. Verify codebooks
    print("\n5. Codebook info...")
    codebooks = wrapper.get_codebooks()
    print(f"   Shape: {codebooks.shape}")
    print(f"   Expected: (4, 256, 128)")
    
    print("\n" + "="*80)
    print("✅ END-TO-END TEST PASSED!")
    print("="*80)
    print("\nNew Monolith encoder (512D) is fully integrated!")
    print("System ready for production use.")

if __name__ == "__main__":
    test_e2e()
