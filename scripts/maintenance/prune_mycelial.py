"""
Prune Mycelial Network - Remove weak connections
Hybrid approach: Keep strong connections, remove noise
"""
import numpy as np
from pathlib import Path

STATE_PATH = Path("data/mycelial_state.npz")
PRUNE_THRESHOLD = 1.0  # Remove all connections with weight < 1.0 (below average)

print("=" * 60)
print("🔪 MYCELIAL NETWORK PRUNING")
print("=" * 60)

# Load state
state = dict(np.load(STATE_PATH, allow_pickle=True))
print(f"\n📊 Before pruning:")
print(f"   Observations: {state['total_observations']:,}")

graph = state['graph'].item() if hasattr(state['graph'], 'item') else state['graph']

# Count before
edges_before = sum(len(n) for n in graph.values())
nodes_before = len(graph)

weights = [w for neighbors in graph.values() for w in neighbors.values()]
print(f"   Nodes: {nodes_before:,}")
print(f"   Edges: {edges_before:,}")
print(f"   Mean weight: {np.mean(weights):.4f}")
print(f"   Max weight: {np.max(weights):.4f}")
print(f"   Density: {edges_before / (nodes_before ** 2):.2%}")

# Prune
print(f"\n🔪 Pruning edges with weight < {PRUNE_THRESHOLD}...")

nodes_to_remove = []
total_pruned = 0

for node_a, neighbors in list(graph.items()):
    neighbors_to_remove = []
    
    for node_b, weight in neighbors.items():
        if weight < PRUNE_THRESHOLD:
            neighbors_to_remove.append(node_b)
    
    for node_b in neighbors_to_remove:
        del neighbors[node_b]
        total_pruned += 1
    
    if not neighbors:
        nodes_to_remove.append(node_a)

for node_a in nodes_to_remove:
    del graph[node_a]

# Count after
edges_after = sum(len(n) for n in graph.values())
nodes_after = len(graph)

weights_after = [w for neighbors in graph.values() for w in neighbors.values()]
print(f"\n📊 After pruning:")
print(f"   Nodes: {nodes_after:,} ({nodes_before - nodes_after:,} removed)")
print(f"   Edges: {edges_after:,} ({total_pruned:,} pruned)")
if nodes_after > 0:
    print(f"   Mean weight: {np.mean(weights_after):.4f}")
    print(f"   Max weight: {np.max(weights_after):.4f}")
    density_after = edges_after / (nodes_after ** 2) if nodes_after > 0 else 0
    print(f"   Density: {density_after:.2%}")

# Save
state['graph'] = graph
np.savez(STATE_PATH, **state)

print(f"\n✅ Saved to {STATE_PATH}")
print(f"   Reduction: {edges_before:,} → {edges_after:,} ({100 * (1 - edges_after/edges_before):.1f}% removed)")
