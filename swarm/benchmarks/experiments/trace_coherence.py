"""
Coherence Analysis: Trace Navigation Steps

Re-runs the massive navigation test but captures the semantic content of each step.
For every step in the path, it finds the nearest real chunks in LanceDB to verify
if the bridge makes sense textually.
"""

import sys
import os
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.memory.storage import LanceDBStorage
from core.topology.topology_engine import create_topology_engine
from swarm.integrations.nemesis import create_swarm_nemesis_integration

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("NAVIGATION COHERENCE TRACE")
print("=" * 60)

# 1. Load Data & Topology
print("\n[1] Loading 20k vectors...")
storage = LanceDBStorage()
df = storage.table.search().limit(20000).to_pandas()
vectors = np.stack(df['vector'].values)
contents = df['content'].tolist()
print(f"✓ Loaded {len(vectors)} chunks")

# 2. Define Start (Average) and Target (Outlier)
centroid = np.mean(vectors, axis=0)
centroid = centroid / np.linalg.norm(centroid)

# Start = Closest to centroid
start_idx = np.argmax(np.dot(vectors, centroid))
start_vec = vectors[start_idx]
start_text = contents[start_idx]

# Target = Furthest from centroid (Outlier)
# We recreate the outlier search logic briefly
outliers = []
for i, vec in enumerate(vectors):
    sim = float(np.dot(centroid, vec))
    outliers.append((i, 1.0 - sim))
outliers.sort(key=lambda x: x[1], reverse=True)
target_idx = outliers[0][0]
target_vec = vectors[target_idx]
target_text = contents[target_idx]

print(f"\n📍 START (Index {start_idx}):\n{start_text[:150]}...")
print(f"\n🎯 TARGET (Index {target_idx}):\n{target_text[:150]}...")

# 3. Navigate
print("\n" + "=" * 60)
print("NAVIGATING...")
print("=" * 60)

topology = create_topology_engine()
integration = create_swarm_nemesis_integration(topology_engine=topology)

result = integration.navigate_and_explain(
    start_concept=start_vec,
    target_concept=target_vec,
    generate_explanation=True
)

path = result.swarm_result.get('path', [])
print(f"✓ Navigation finished in {len(path)} steps")

# 4. Trace the Path (Find nearest chunks for each step)
print("\n" + "=" * 60)
print("PATH ANALYSIS (Step-by-Step Content)")
print("=" * 60)

for i, pos in enumerate(path):
    # Find nearest chunk in the database to this position
    sims = np.dot(vectors, pos)
    nearest_idx = np.argmax(sims)
    nearest_text = contents[nearest_idx]
    nearest_sim = sims[nearest_idx]
    
    print(f"\n👣 STEP {i}")
    print(f"   Sim to pos: {nearest_sim:.3f}")
    print(f"   Content: \"{nearest_text[:200]}...\"")
    
    # Check coherence with previous
    if i > 0:
        prev_pos = path[i-1]
        step_sim = np.dot(pos, prev_pos)
        print(f"   (Proximity to previous step: {step_sim:.3f})")

print("\n" + "=" * 60)
print("TRACE COMPLETE")
print("=" * 60)
