"""Quick mycelial health check."""
import numpy as np

state = np.load('data/mycelial_state.npz', allow_pickle=True)

print("=" * 50)
print("🍄 MYCELIAL NETWORK HEALTH")
print("=" * 50)

obs = state['total_observations']
step = state['step']
graph = state['graph'].item() if hasattr(state['graph'], 'item') else state['graph']

num_nodes = len(graph)
num_edges = sum(len(neighbors) for neighbors in graph.values())

weights = [w for neighbors in graph.values() for w in neighbors.values()]
mean_weight = sum(weights) / len(weights) if weights else 0
max_weight = max(weights) if weights else 0

print(f"\n📊 Statistics:")
print(f"   Total Observations: {obs:,}")
print(f"   Training Steps: {step:,}")
print(f"   Active Nodes: {num_nodes:,}")
print(f"   Active Edges: {num_edges:,}")
print(f"   Mean Weight: {mean_weight:.4f}")
print(f"   Max Weight: {max_weight:.4f}")

if num_nodes > 0:
    density = num_edges / (num_nodes * num_nodes)
    print(f"   Density: {density:.4%}")

print(f"\n❤️ Health Assessment:")
print(f"   Has connections: {'✅' if num_edges > 0 else '❌'}")
print(f"   Sufficient observations (>10k): {'✅' if obs > 10000 else '❌'}")
print(f"   Sparse (<10%): {'✅' if (num_nodes == 0 or num_edges / (num_nodes * num_nodes) < 0.1) else '❌'}")
print(f"   Strong max weight (>0.5): {'✅' if max_weight > 0.5 else '⚠️'}")
