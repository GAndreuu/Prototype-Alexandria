"""
Test: Geodesic vs Euclidean Bridge (PCA-Reduced Version)
=========================================================
Uses PCA to reduce 384D → 32D for tractable geodesic computation.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import os
from sklearn.decomposition import PCA

def main():
    print("="*70)
    print("TEST: GEODESIC VS EUCLIDEAN BRIDGE (PCA-32D)")
    print("="*70)
    
    # Load components
    from core.topology.topology_engine import TopologyEngine
    from core.memory.storage import LanceDBStorage
    from core.field.manifold import DynamicManifold, ManifoldConfig
    from core.field.metric import RiemannianMetric, MetricConfig
    from core.field.geodesic_flow import GeodesicFlow, GeodesicConfig
    
    print("\n[1] Loading components...")
    
    topology = TopologyEngine()
    storage = LanceDBStorage()
    
    # Test pairs
    test_pairs = [
        ("active inference", "thermodynamics"),
        ("active inference", "variational autoencoder"),
        ("predictive coding", "reinforcement learning"),
    ]
    
    print("\n[2] Loading embeddings...")
    
    # Get embeddings
    term_embeddings = {}
    for term1, term2 in test_pairs:
        for term in [term1, term2]:
            if term not in term_embeddings:
                emb = topology.encode([term])[0]
                term_embeddings[term] = emb
    
    # Stack all test embeddings
    all_terms = list(term_embeddings.keys())
    test_embs = np.array([term_embeddings[t] for t in all_terms])
    
    print(f"   Original dimension: {test_embs.shape[1]}")
    print(f"   Test terms: {len(all_terms)}")
    
    # Load training embeddings to fit PCA (need more samples than components)
    print("\n[3] Loading training embeddings for PCA...")
    training_embs = np.load("data/training_embeddings.npy")[:1000]  # Use 1000 samples
    print(f"   Training samples: {len(training_embs)}")
    
    # PCA reduce to 32D (32^4 = ~1M ops por passo)
    print("\n[4] Reducing to 32D via PCA...")
    pca = PCA(n_components=32)
    pca.fit(training_embs)  # Fit on training data
    reduced_embs = pca.transform(test_embs)  # Transform test terms
    print(f"   Reduced dimension: {reduced_embs.shape[1]}")
    print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    
    # Create term -> reduced embedding mapping
    reduced_embeddings = {t: reduced_embs[i] for i, t in enumerate(all_terms)}
    
    # Create Field components at 32D
    print("\n[5] Creating 32D Field components...")
    
    manifold_config = ManifoldConfig(base_dim=32, max_expansion=48)
    manifold = DynamicManifold(manifold_config)
    
    metric_config = MetricConfig(deformation_radius=0.5, deformation_strength=0.8)
    metric = RiemannianMetric(manifold, metric_config)
    
    flow_config = GeodesicConfig(max_steps=100, step_size=0.05, use_scipy_integrator=False, n_workers=16)
    flow = GeodesicFlow(manifold, metric, flow_config)
    
    # Deform metric at each term location
    print("\n[6] Deforming metric...")
    for term, emb in reduced_embeddings.items():
        metric.deform_at(emb, intensity=1.5)
    
    print(f"   {len(metric.deformations)} deformations added")
    
    # Test each pair
    print("\n[7] Computing bridges...")
    print("="*70)
    
    results = []
    
    for term1, term2 in test_pairs:
        print(f"\n▶ {term1.upper()} <-> {term2.upper()}")
        
        emb1 = reduced_embeddings[term1]
        emb2 = reduced_embeddings[term2]
        
        # Euclidean
        euclidean_mid = (emb1 + emb2) / 2
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        # Project euclidean midpoint back to 384D for LanceDB search
        euclidean_mid_384 = pca.inverse_transform(euclidean_mid.reshape(1, -1))[0]
        euclidean_results = storage.search(euclidean_mid_384.tolist(), limit=1)
        euclidean_bridge = euclidean_results[0].get('source', '?').split('\\')[-1][:50] if euclidean_results else "none"
        
        # Geodesic
        try:
            geodesic_path = flow.shortest_path(emb1, emb2)
            geodesic_length = geodesic_path.length
            n_points = len(geodesic_path.points)
            
            # Get midpoint
            mid_idx = len(geodesic_path.points) // 2
            geodesic_mid = geodesic_path.points[mid_idx]
            
            # Project back to 384D
            geodesic_mid_384 = pca.inverse_transform(geodesic_mid.reshape(1, -1))[0]
            geodesic_results = storage.search(geodesic_mid_384.tolist(), limit=1)
            geodesic_bridge = geodesic_results[0].get('source', '?').split('\\')[-1][:50] if geodesic_results else "none"
            
            # Difference between midpoints
            midpoint_diff = np.linalg.norm(euclidean_mid - geodesic_mid)
            
        except Exception as e:
            print(f"   ERROR: {e}")
            geodesic_bridge = "ERROR"
            geodesic_length = 0
            n_points = 0
            midpoint_diff = 0
        
        # Riemannian distance
        try:
            riemannian_dist = metric.distance(emb1, emb2)
        except:
            riemannian_dist = euclidean_dist
        
        # Curvature at midpoint
        try:
            curvature = metric.curvature_scalar_at(euclidean_mid)
        except:
            curvature = 0
        
        print(f"\n   Euclidean distance:  {euclidean_dist:.3f}")
        print(f"   Riemannian distance: {riemannian_dist:.3f}")
        print(f"   Geodesic length:     {geodesic_length:.3f} ({n_points} points)")
        print(f"   Midpoint difference: {midpoint_diff:.4f}")
        print(f"   Curvature at mid:    {curvature:.4f}")
        
        print(f"\n   EUCLIDEAN BRIDGE: {euclidean_bridge}")
        print(f"   GEODESIC BRIDGE:  {geodesic_bridge}")
        
        different = euclidean_bridge != geodesic_bridge
        print(f"\n   {'✓ DIFFERENT bridges!' if different else '= Same bridge'}")
        
        results.append({
            'pair': (term1, term2),
            'euclidean_dist': euclidean_dist,
            'riemannian_dist': riemannian_dist,
            'geodesic_length': geodesic_length,
            'midpoint_diff': midpoint_diff,
            'different_bridge': different
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n_different = sum(1 for r in results if r['different_bridge'])
    avg_diff = np.mean([r['midpoint_diff'] for r in results])
    
    print(f"\nPairs with different bridges: {n_different}/{len(results)}")
    print(f"Average midpoint deviation:   {avg_diff:.4f}")
    
    # Check if riemannian differs from euclidean
    for r in results:
        ratio = r['riemannian_dist'] / r['euclidean_dist'] if r['euclidean_dist'] > 0 else 1
        if abs(ratio - 1) > 0.01:
            print(f"\n✓ Metric is curved! Riemannian/Euclidean ratio: {ratio:.2f}")
            break
    else:
        print("\n= Metric appears flat (Riemannian ≈ Euclidean)")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
    If midpoint_diff > 0.1  → geodesic deviates from straight line
    If bridges differ       → Field finds different connection path  
    If Riemannian ≠ Euclid  → metric is successfully deformed
    """)


if __name__ == "__main__":
    main()
