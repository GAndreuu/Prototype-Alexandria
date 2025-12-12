"""
Alexandria System Stress Test
Test system performance under heavy load.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import psutil
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from core.reasoning.neural_learner import V2Learner
from core.reasoning.mycelial_reasoning import MycelialReasoning
from core.memory.storage import LanceDBStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def stress_test_vqvae(learner, num_iterations=1000):
    """Test VQ-VAE encoding/decoding speed"""
    logger.info(f"\n{'='*60}")
    logger.info("🧠 VQ-VAE Stress Test")
    logger.info(f"{'='*60}")
    
    # Generate random vectors (simulating embeddings)
    vectors = np.random.randn(num_iterations, 384).astype(np.float32).tolist()
    
    mem_before = measure_memory()
    start = time.time()
    
    # Encode all vectors
    for i, vec in enumerate(vectors):
        learner.encode([vec])
        if (i + 1) % 100 == 0:
            logger.info(f"Encoded {i+1}/{num_iterations} vectors...")
    
    elapsed = time.time() - start
    mem_after = measure_memory()
    
    ops_per_sec = num_iterations / elapsed
    
    logger.info(f"✅ Encoded {num_iterations} vectors in {elapsed:.2f}s")
    logger.info(f"   Throughput: {ops_per_sec:.1f} ops/sec")
    logger.info(f"   Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (Δ {mem_after-mem_before:.1f} MB)")
    
    return {
        'operations': num_iterations,
        'time': elapsed,
        'ops_per_sec': ops_per_sec,
        'memory_delta': mem_after - mem_before
    }

def stress_test_mycelial(mycelial, num_observations=10000):
    """Test Mycelial network observation capacity"""
    logger.info(f"\n{'='*60}")
    logger.info("🍄 Mycelial Network Stress Test")
    logger.info(f"{'='*60}")
    
    mem_before = measure_memory()
    start = time.time()
    
    # Generate random indices (4 heads, 256 codes each)
    for i in range(num_observations):
        indices = np.random.randint(0, 256, size=4)
        mycelial.observe(indices)
        
        if (i + 1) % 1000 == 0:
            stats = mycelial.get_network_stats()
            logger.info(f"Observations: {i+1}/{num_observations} | Connections: {stats['active_connections']} | Density: {stats['density']:.6f}")
    
    elapsed = time.time() - start
    mem_after = measure_memory()
    
    stats = mycelial.get_network_stats()
    
    logger.info(f"✅ {num_observations} observations processed in {elapsed:.2f}s")
    logger.info(f"   Throughput: {num_observations/elapsed:.1f} obs/sec")
    logger.info(f"   Active Connections: {stats['active_connections']}")
    logger.info(f"   Network Density: {stats['density']:.6f}")
    logger.info(f"   Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (Δ {mem_after-mem_before:.1f} MB)")
    
    return {
        'observations': num_observations,
        'time': elapsed,
        'obs_per_sec': num_observations / elapsed,
        'connections': stats['active_connections'],
        'density': stats['density'],
        'memory_delta': mem_after - mem_before
    }

def stress_test_lancedb(storage, num_queries=1000):
    """Test LanceDB query performance"""
    logger.info(f"\n{'='*60}")
    logger.info("💾 LanceDB Query Stress Test")
    logger.info(f"{'='*60}")
    
    # Generate random query vectors
    query_vectors = np.random.randn(num_queries, 384).astype(np.float32).tolist()
    
    mem_before = measure_memory()
    start = time.time()
    
    for i, vec in enumerate(query_vectors):
        results = storage.search(vec, limit=10)
        if (i + 1) % 100 == 0:
            logger.info(f"Queries: {i+1}/{num_queries}")
    
    elapsed = time.time() - start
    mem_after = measure_memory()
    
    queries_per_sec = num_queries / elapsed
    
    logger.info(f"✅ {num_queries} queries completed in {elapsed:.2f}s")
    logger.info(f"   Throughput: {queries_per_sec:.1f} queries/sec")
    logger.info(f"   Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (Δ {mem_after-mem_before:.1f} MB)")
    
    return {
        'queries': num_queries,
        'time': elapsed,
        'queries_per_sec': queries_per_sec,
        'memory_delta': mem_after - mem_before
    }

def main():
    logger.info("🚀 Starting Alexandria System Stress Test")
    logger.info(f"Initial Memory: {measure_memory():.1f} MB\n")
    
    results = {}
    
    # 1. VQ-VAE Test
    try:
        learner = V2Learner()
        results['vqvae'] = stress_test_vqvae(learner, num_iterations=1000)
    except Exception as e:
        logger.error(f"VQ-VAE test failed: {e}")
        results['vqvae'] = {'error': str(e)}
    
    # 2. Mycelial Test
    try:
        mycelial = MycelialReasoning()
        results['mycelial'] = stress_test_mycelial(mycelial, num_observations=10000)
    except Exception as e:
        logger.error(f"Mycelial test failed: {e}")
        results['mycelial'] = {'error': str(e)}
    
    # 3. LanceDB Test
    try:
        storage = LanceDBStorage()
        results['lancedb'] = stress_test_lancedb(storage, num_queries=1000)
    except Exception as e:
        logger.error(f"LanceDB test failed: {e}")
        results['lancedb'] = {'error': str(e)}
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 STRESS TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    if 'vqvae' in results and 'ops_per_sec' in results['vqvae']:
        logger.info(f"VQ-VAE:        {results['vqvae']['ops_per_sec']:.1f} ops/sec")
    
    if 'mycelial' in results and 'obs_per_sec' in results['mycelial']:
        logger.info(f"Mycelial:      {results['mycelial']['obs_per_sec']:.1f} obs/sec")
        logger.info(f"  └─ Connections: {results['mycelial']['connections']}")
    
    if 'lancedb' in results and 'queries_per_sec' in results['lancedb']:
        logger.info(f"LanceDB:       {results['lancedb']['queries_per_sec']:.1f} queries/sec")
    
    logger.info(f"\nFinal Memory: {measure_memory():.1f} MB")
    logger.info("✅ Stress test complete!")

if __name__ == "__main__":
    main()
