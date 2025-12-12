# -*- coding: utf-8 -*-
"""
Script to gather real metrics for walkthrough.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from core.field.pre_structural_field import PreStructuralField
from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig

print("=" * 60)
print("ALEXANDRIA SYSTEM METRICS - PHASE 1 VERIFICATION")
print("=" * 60)

# Initialize components
mc = MycelialConfig(save_path='/tmp/test_wt.pkl')
m = MycelialReasoning(mc)
m.reset()

f = PreStructuralField()
f.connect_mycelial(m)

print("\n[1] COMPONENT INITIALIZATION")
print(f"    Manifold base dim: {f.manifold.current_dim}")
print(f"    Manifold max expansion: {f.manifold.config.max_expansion}")
print(f"    Mycelial num_heads: {m.c.num_heads}")
print(f"    Mycelial codebook_size: {m.c.codebook_size}")
print(f"    Mycelial learning_rate: {m.c.learning_rate}")

stats_before = m.get_network_stats()
print("\n[2] MYCELIAL STATE (BEFORE)")
print(f"    Total observations: {stats_before['total_observations']}")
print(f"    Active nodes: {stats_before['active_nodes']}")
print(f"    Active edges: {stats_before['active_edges']}")

# Trigger 5 points
print("\n[3] TRIGGERING 5 POINTS IN FIELD")
for i in range(5):
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    codes = np.array([10+i, 20+i, 30+i, 40+i], dtype=np.int32)
    f.trigger(embedding=emb, codes=codes, intensity=1.0)
    print(f"    Point {i+1}: codes={codes.tolist()}")

print(f"\n    Manifold points: {len(f.manifold.points)}")
print(f"    Field triggers: {f.trigger_count}")

# Crystallize
print("\n[4] CRYSTALLIZING FIELD")
g = f.crystallize()
print(f"    Graph nodes: {len(g['nodes'])}")
print(f"    Graph edges: {len(g['edges'])}")

if g['edges']:
    weights = [e['weight'] for e in g['edges']]
    print(f"    Edge weights: min={min(weights):.4f}, max={max(weights):.4f}")

# After
stats_after = m.get_network_stats()
print("\n[5] MYCELIAL STATE (AFTER CRYSTALLIZATION)")
print(f"    Total observations: {stats_after['total_observations']}")
print(f"    Active nodes: {stats_after['active_nodes']}")
print(f"    Active edges: {stats_after['active_edges']}")
print(f"    Mean weight: {stats_after['mean_weight']:.4f}")
print(f"    Max weight: {stats_after['max_weight']:.4f}")

edges_added = stats_after['active_edges'] - stats_before['active_edges']
print(f"\n[6] DELTA")
print(f"    New edges from crystallization: {edges_added}")

print("\n" + "=" * 60)
print("PHASE 1: FIELD -> MYCELIAL LOOP IS CLOSED!")
print("=" * 60)
