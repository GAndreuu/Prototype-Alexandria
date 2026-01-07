"""
Massive Test: 20k Papers Analysis

This script performs a stress test and "gold finding" operation on the 
full 20k+ paper dataset stored in LanceDB.

Capabilities tested:
1. Data Loading: Retrieving 20k vectors from LanceDB
2. Outlier Detection: Finding unique papers in the massive dataset
3. Active Bridge: Connecting distant concepts across the 20k corpus
"""

import sys
import os
import time
import numpy as np
import logging

# Setup paths
# Current path: swarm/benchmarks/experiments/massive_test.py
# Root path: ../../../
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Explicit imports to avoid ambiguity
from core.memory.storage import LanceDBStorage
from core.topology.topology_engine import create_topology_engine
from swarm.integrations.nemesis import create_swarm_nemesis_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MassiveTest")

print("=" * 60)
print("ALEXANDRIA SWARM: MASSIVE DATASET TEST (20k Papers)")
print("=" * 60)

# 1. Connect and Load Data
print("\n[1] Connecting to LanceDB...")
try:
    storage = LanceDBStorage()
    row_count = storage.count()
    print(f"✓ Connected to LanceDB. Total rows: {row_count}")
    
    if row_count < 100:
        print("⚠️ Warning: Database seems empty or small. Generating mock data for test...")
        # (Generation logic could go here, but let's assume user has data)
        USE_MOCK = True
    else:
        USE_MOCK = False
        
except Exception as e:
    print(f"❌ Failed to connect to LanceDB: {e}")
    sys.exit(1)

# 2. Retrieve Vectors
print(f"\n[2] Retrieving vectors (Limit: 20,000)...")
start_time = time.time()

if not USE_MOCK:
    # Use PyArrow to fetch data efficiently without pandas
    table_arrow = storage.table.search().limit(20000).to_arrow()
    
    ids = table_arrow["id"].to_pylist()
    contents = table_arrow["content"].to_pylist()
    # vectors column is a FixedSizeListArray or ListArray, manual conversion might be needed
    # Flatten/stack logic depending on arrow type
    vectors_list = table_arrow["vector"].to_pylist() 
    vectors = np.array(vectors_list)
    
else:
    # Generate 1000 mock vectors associated with AI topics
    print("   Generating 1000 mock vectors...")
    vectors = np.random.randn(1000, 384)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"mock_{i}" for i in range(1000)]
    contents = [f"Mock Paper {i} about AI and Swarms" for i in range(1000)]
    
    # Inject some Gold
    vectors[50] = vectors[0] * 0.5 + np.random.randn(384) * 0.5 # Outlier
    contents[50] = "★ GOLD: Novel approach to Artificial Consciousness"

print(f"✓ Loaded {len(vectors)} vectors in {time.time() - start_time:.2f}s")
print(f"   Shape: {vectors.shape}")

# 3. Initialize Topology Engine
print("\n[3] Initializing Topology Engine...")
topology = create_topology_engine()

# 4. Find Outliers (Gold Mining)
print("\n[4] ⛏️  Mining for GOLD (Outlier Detection)...")
print("   Analyzing 20k papers to find unique conceptual contributions...")
start_time = time.time()

# We pass the loaded vectors to find_outliers
# Note: find_outliers expects (embeddings, labels, top_k)
outliers = topology.find_outliers(vectors, labels=contents, top_k=5)

duration = time.time() - start_time
print(f"✓ Analysis complete in {duration:.2f}s")

print("\n🏆 TOP 5 UNIQUE PAPERS (GOLD):")
for i, (emb, dist, label) in enumerate(outliers):
    preview = label[:100] + "..." if len(label) > 100 else label
    print(f"   {i+1}. [Dist: {dist:.4f}] {preview}")

# 5. Massive Navigation
print("\n[5] 🧭 Massive Navigation Test")
print("   Attempting to bridge the most distant concepts found...")

# Pick start (most generic/centroid-like) and target (most outlier)
# Centroid is simple mean
centroid = np.mean(vectors, axis=0)
centroid = centroid / np.linalg.norm(centroid)

# Find vector closest to centroid (Most standard paper)
sims = np.dot(vectors, centroid)
most_standard_idx = np.argmax(sims)
start_paper = contents[most_standard_idx]
start_vec = vectors[most_standard_idx]

# Target is the top outlier
target_paper = outliers[0][2]
target_vec = outliers[0][0]

print(f"   Start: {start_paper[:60]}...")
print(f"   Target: {target_paper[:60]}...")

# Init Integration
integration = create_swarm_nemesis_integration(topology_engine=topology)

# Inject the REAL data into the navigator's topology context if possible
# For now, the navigator uses the topology's internal cluster centers or loads from file.
# To make it use the 20k vectors, we might need to update the topology engine's knowledge,
# but for this test, we'll let it use its internal map + the start/target vectors.

result = integration.navigate_and_explain(
    start_concept=start_vec, # Pass vector directly
    target_concept=target_vec, # Pass vector directly
    generate_explanation=True
)

print("\n   Navigation Result:")
print(f"   {result.explanation.summary}")
print(f"   {result.explanation.path_description}")
print(f"   {result.explanation.neurotype_narrative}")

print("\n" + "=" * 60)
print("MASSIVE TEST COMPLETE")
print("=" * 60)
