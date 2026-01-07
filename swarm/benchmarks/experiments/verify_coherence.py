"""
Semantic Coherence Verification Test

Verifica se os embeddings do TopologyEngine fazem sentido semântico.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

print("=" * 60)
print("SEMANTIC COHERENCE VERIFICATION")
print("=" * 60)

# Load topology
from core.topology.topology_engine import create_topology_engine
topology = create_topology_engine()

# Test pairs with EXPECTED similarities
test_pairs = [
    # (concept1, concept2, expected_min, expected_max, reasoning)
    ("dog", "cat", 0.4, 0.9, "Ambos são animais domésticos"),
    ("dog", "airplane", 0.0, 0.3, "Sem relação semântica"),
    ("machine learning", "deep learning", 0.5, 1.0, "Deep learning é subtipo"),
    ("neural network", "brain", 0.3, 0.8, "Inspiração biológica"),
    ("python", "programming", 0.3, 0.8, "Python é linguagem de programação"),
    ("car", "automobile", 0.7, 1.0, "Sinônimos"),
    ("happy", "sad", 0.2, 0.6, "Opostos mas mesmo campo semântico"),
    ("quantum physics", "recipe cooking", 0.0, 0.25, "Campos completamente diferentes"),
]

print("\nGenerating embeddings...")
all_concepts = []
for c1, c2, _, _, _ in test_pairs:
    if c1 not in all_concepts:
        all_concepts.append(c1)
    if c2 not in all_concepts:
        all_concepts.append(c2)

embeddings = topology.encode(all_concepts)
emb_dict = {c: embeddings[i] for i, c in enumerate(all_concepts)}

print(f"Generated {len(all_concepts)} embeddings")
print("\n" + "=" * 60)
print("SIMILARITY TESTS")
print("=" * 60)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

passed = 0
failed = 0

for c1, c2, expected_min, expected_max, reasoning in test_pairs:
    sim = cosine_sim(emb_dict[c1], emb_dict[c2])
    
    in_range = expected_min <= sim <= expected_max
    
    if in_range:
        status = "✅ PASS"
        passed += 1
    else:
        status = "❌ FAIL"
        failed += 1
    
    print(f"\n{status}: '{c1}' ↔ '{c2}'")
    print(f"  Similarity: {sim:.3f}")
    print(f"  Expected: [{expected_min:.2f}, {expected_max:.2f}]")
    print(f"  Reason: {reasoning}")

print("\n" + "=" * 60)
print(f"TOTAL: {passed}/{passed+failed} tests passed")
print("=" * 60)

if failed == 0:
    print("\n🎉 Embeddings are semantically coherent!")
else:
    print(f"\n⚠️  {failed} tests failed - check embedding model")
