"""
Alexandria Experiment: kNN vs Geodesic Bridge
=============================================

Este script executa um experimento comparativo para validar a hipótese de que
a Geodesic Bridge encontra conexões semânticas mais ricas que o baseline Euclidiano.

Setup:
- 10 Queries simuladas (Start -> End).
- Espaço Latente Sintético (128d) populado com "Papers".
- Atratores injetados para curvar o espaço (simulando papers seminais/hubs).

Métodos:
1. Baseline: Interpolação Linear + kNN (Euclidiano "Flat").
2. Challenger: Geodesic Bridge + kNN (Riemanniano "Curved").

Métricas:
- Ratio (Geodesic/Euclidean Length)
- Attractor Hits (Quantos atratores foram "descobertos" no caminho?)
- Overlap (Jaccard entre conjuntos de papers recuperados)
- Serendipity (Papers encontrados apenas pelo Geodesic)

Autor: Alexandria System
"""

import os
import sys
import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set

# Config logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Experiment")

# Add root to path
sys.path.append(os.getcwd())

try:
    from core.integrations.geodesic_bridge_integration import GeodesicBridgeIntegration, GeodesicBridgeConfig
    from core.field.manifold import ManifoldConfig
except ImportError:
    logger.error("Failed to import core modules. Run from project root.")
    sys.exit(1)

# =============================================================================
# MOCK ENVIRONMENT
# =============================================================================

@dataclass
class Paper:
    id: int
    title: str
    embedding: np.ndarray
    is_attractor: bool = False

class SyntheticKnowledgeBase:
    """Simula uma base de conhecimento indexada (Vector DB)."""
    
    def __init__(self, n_papers: int = 500, dim: int = 128, n_attractors: int = 20):
        self.dim = dim
        self.papers: List[Paper] = []
        
        logger.info(f"Generating synthetic KB with {n_papers} papers ({n_attractors} attractors)...")
        
        # 1. Generate regular papers (Background noise/clusters)
        # Cluster centers
        n_clusters = 5
        centers = np.random.randn(n_clusters, dim)
        
        for i in range(n_papers):
            # Assign to random cluster to create structure
            c_idx = np.random.randint(0, n_clusters)
            center = centers[c_idx]
            noise = np.random.randn(dim) * 0.5
            emb = center + noise
            emb = emb / np.linalg.norm(emb) # Normalize
            
            self.papers.append(Paper(i, f"Paper_{i}", emb))
            
        # 2. Promote random papers to Attractors (Hubs)
        # Atratores são papers que exercem gravidade (conceitos fundamentais)
        perm = np.random.permutation(n_papers)
        attractor_indices = perm[:n_attractors]
        
        self.attractors = []
        for idx in attractor_indices:
            self.papers[idx].is_attractor = True
            self.papers[idx].title = f"ATTRACTOR_{idx}"
            self.attractors.append(self.papers[idx])
            
        # Index for kNN (Brute force is fine for 500 items)
        self.embeddings = np.array([p.embedding for p in self.papers])
        
    def get_nearest(self, query: np.ndarray, k: int = 5) -> List[Tuple[Paper, float]]:
        """Busca kNN Euclidiano."""
        dists = np.linalg.norm(self.embeddings - query, axis=1)
        indices = np.argsort(dists)[:k]
        return [(self.papers[i], dists[i]) for i in indices]

class ExperimentBridge:
    """Adaptador para o GeodesicBridgeIntegration."""
    
    def __init__(self, kb: SyntheticKnowledgeBase):
        self.kb = kb
        self.latent_dim = kb.dim
        # Expose attractors as anchors for the metric construction
        # Metric expects objects with 'global_idx' and 'coordinates'
        self.anchors = [
            type('Anchor', (), {'global_idx': p.id, 'coordinates': p.embedding, 'activation': 1.0})() 
            for p in kb.attractors
        ]
        
    def _project_to_latent(self, x):
        return x
        
    def get_nearest_anchors(self, point, k=1):
        # Used by propagation/finding nearby attractors
        # Return list of (Anchor, dist)
        # Simula busca apenas nos atratores (para a métrica saber onde eles estão)
        dists = []
        for anchor in self.anchors:
            d = float(np.linalg.norm(point - anchor.coordinates))
            dists.append((anchor, d))
        dists.sort(key=lambda x: x[1])
        return dists[:k]

# =============================================================================
# EXPERIMENT ENGINE
# =============================================================================

def run_experiment():
    print("="*60)
    print("ALEXANDRIA EXPERIMENT: kNN vs GEODESIC")
    print("="*60)
    
    # 1. Setup
    # USAR DIMENSÃO MENOR PARA O EXPERIMENTO
    # Isso garante que active_dims venha a cobrir todo o espaço, 
    # permitindo que a física "veja" os atratores e rode rápido.
    DIM = 32 
    kb = SyntheticKnowledgeBase(n_papers=500, dim=DIM, n_attractors=15)
    bridge_adapter = ExperimentBridge(kb)
    
    # Configurar Integration
    config = GeodesicBridgeConfig(
        max_geodesic_steps=100,
        geodesic_step_size=0.1, # Step um pouco maior
        use_attractor_guidance=True,
        attractor_pull_strength=5.0, # Muito forte para garantir efeito visual/numérico
        geodesic_tolerance=1e-1
    )
    
    gbi = GeodesicBridgeIntegration(bridge_adapter, config=config)
    # Hack para forçar active_dims no flow criado internamente
    if gbi.geodesic:
        gbi.geodesic.config.active_dims = DIM 
        gbi.geodesic.config.dt = 0.05
        # Relaxar shooting
        gbi.geodesic.config.shooting_iters = 15
        gbi.geodesic.config.tol = 0.1
    
    # Injeter deformações na métrica explicitamente (já que o Adapter mocka anchors)
    # Na integração real, isso acontece dinamicamente. Aqui forçamos o setup estático.
    print("Injecting attractors into Metric...")
    for anchor in bridge_adapter.anchors:
        # Deformar métrica na posição do atrator
        gbi.geodesic.metric.deform_at(
            anchor.coordinates, 
            intensity=0.8, # Intensidade alta
            radius=1.5     # Raio médio
        )
        
    # 2. Queries
    n_queries = 5
    results = []
    
    print(f"\nRunning {n_queries} comparative queries...", flush=True)
    print("-" * 60, flush=True)
    print(f"{'ID':<4} | {'Start->End Dist':<15} | {'Geo Ratio':<10} | {'kNN Hits':<10} | {'Geo Hits':<10} | {'Serendipity':<11}", flush=True)
    print("-" * 60, flush=True)
    
    for i in range(n_queries):
        # Pick random start and end (far enough apart)
        start_idx, end_idx = np.random.choice(len(kb.papers), 2, replace=False)
        start_paper = kb.papers[start_idx]
        end_paper = kb.papers[end_idx]
        
        while np.linalg.norm(start_paper.embedding - end_paper.embedding) < 1.0:
             # Retry if too close
             start_idx, end_idx = np.random.choice(len(kb.papers), 2, replace=False)
             start_paper = kb.papers[start_idx]
             end_paper = kb.papers[end_idx]
             
        # A. Baseline: Linear Path + kNN sampling
        dist = np.linalg.norm(start_paper.embedding - end_paper.embedding)
        linear_papers = set()
        steps = 10
        for t in np.linspace(0, 1, steps):
            pt = start_paper.embedding * (1-t) + end_paper.embedding * t
            neighbors = kb.get_nearest(pt, k=3)
            for p, d in neighbors:
                linear_papers.add(p.id)
                
        # B. Challenger: Geodesic Path + kNN sampling
        geo_result = gbi.semantic_path(start_paper.embedding, end_paper.embedding)
        
        geo_papers = set()
        # Amostrar ao longo do caminho geodésico
        if geo_result.points is not None and len(geo_result.points) > 0:
            # Subsample points
            path_pts = geo_result.points[::max(1, len(geo_result.points)//steps)]
            for pt in path_pts:
                 neighbors = kb.get_nearest(pt, k=3)
                 for p, d in neighbors:
                    geo_papers.add(p.id)
        
        # Metrics
        # Atratores encontrados
        knn_attractors = sum(1 for pid in linear_papers if kb.papers[pid].is_attractor)
        geo_attractors = sum(1 for pid in geo_papers if kb.papers[pid].is_attractor)
        
        # Serendipity: Papers no Geo que NÃO estão no Linear e são Atratores
        # (Ou simplesmente papers únicos relevantes)
        unique_geo = geo_papers - linear_papers
        serendipity_count = len(unique_geo)
        
        # Ratio
        ratio = geo_result.geodesic_ratio
        
        print(f"{i:<4} | {dist:<15.4f} | {ratio:<10.4f} | {knn_attractors:<10} | {geo_attractors:<10} | {serendipity_count:<11}", flush=True)
        
        results.append({
            "query_id": i,
            "dist": dist,
            "ratio": ratio,
            "knn_hits": knn_attractors,
            "geo_hits": geo_attractors,
            "serendipity": serendipity_count
        })
        
    # 3. Summary
    print("-" * 60)
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_knn = np.mean([r['knn_hits'] for r in results])
    avg_geo = np.mean([r['geo_hits'] for r in results])
    
    print("\nRESULTS SUMMARY:")
    print(f"Average Geodesic Ratio: {avg_ratio:.4f} (Expect < 1.0 if shortcuts found, or > 1.0 if avoiding obstacles/detouring to attractors)")
    print(f"Average Attractor Hits (kNN): {avg_knn:.2f}")
    print(f"Average Attractor Hits (Geo): {avg_geo:.2f}")
    
    if avg_geo > avg_knn:
        print("\n✅ SUCCESS: Geodesic Bridge found more semantic hubs (attractors) than linear baseline.")
    else:
        print("\n❌ FAILURE: Geodesic Bridge did not outperform baseline.")

if __name__ == "__main__":
    run_experiment()
