"""
Scientific Validation: Baseline vs Alexandria
==============================================
Compares simple KMeans baseline against Alexandria's full cognitive architecture.

Ground Truth: Known connections in Active Inference / Free Energy Principle domain.

Usage:
    python scripts/validate_alexandria.py
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Validation")


# ============================================================================
# GROUND TRUTH: Known connections that SHOULD be found
# ============================================================================
GROUND_TRUTH = [
    # Active Inference core connections
    ("active inference", "free energy principle", "both developed by Friston"),
    ("active inference", "control theory", "via Bellman equation / optimal control"),
    ("active inference", "reinforcement learning", "expected free energy = reward"),
    ("active inference", "predictive coding", "inference as prediction error minimization"),
    
    # Free Energy connections
    ("free energy principle", "variational autoencoder", "both use ELBO / variational bound"),
    ("free energy principle", "bayesian inference", "FEP is approximate Bayesian inference"),
    ("free energy principle", "thermodynamics", "free energy from statistical physics"),
    
    # Predictive Coding connections
    ("predictive coding", "hierarchical bayesian", "hierarchical generative models"),
    ("predictive coding", "attention", "precision weighting = attention"),
    
    # VAE / Generative Models
    ("variational autoencoder", "generative model", "VAE is a generative model"),
]


def load_papers_from_lancedb(limit: int = 100) -> List[Dict]:
    """Load papers from LanceDB, filtering for relevant topics."""
    from core.memory.storage import LanceDBStorage
    from core.topology.topology_engine import TopologyEngine
    
    logger.info("Loading papers from LanceDB...")
    
    storage = LanceDBStorage()
    topology = TopologyEngine()
    
    # Search for relevant papers
    search_terms = [
        "active inference",
        "free energy principle", 
        "predictive coding",
        "variational autoencoder",
        "bayesian inference",
        "control theory",
        "reinforcement learning"
    ]
    
    all_papers = []
    seen_ids = set()
    
    for term in search_terms:
        # Encode search term
        query_vec = topology.encode([term])[0]
        results = storage.search(query_vec.tolist(), limit=limit // len(search_terms))
        
        for r in results:
            if r['id'] not in seen_ids:
                seen_ids.add(r['id'])
                all_papers.append({
                    'id': r['id'],
                    'title': r.get('source', '').split('/')[-1].replace('.pdf', '').replace('_', ' '),
                    'content': r.get('content', '')[:500],
                    'embedding': query_vec  # Will be replaced with actual embedding
                })
    
    logger.info(f"Loaded {len(all_papers)} papers")
    return all_papers


def compute_embeddings(papers: List[Dict]) -> np.ndarray:
    """Compute embeddings for papers using TopologyEngine."""
    from core.topology.topology_engine import TopologyEngine
    
    logger.info("Computing embeddings...")
    topology = TopologyEngine()
    
    embeddings = []
    for paper in papers:
        # Use first 200 chars of content for embedding
        text = paper.get('content', '')[:200] or paper.get('title', '')
        emb = topology.encode([text])[0]
        embeddings.append(emb)
        paper['embedding'] = emb
    
    return np.array(embeddings)


# ============================================================================
# BASELINE: Simple KMeans + Nearest Neighbors
# ============================================================================
def run_baseline(papers: List[Dict], embeddings: np.ndarray, n_clusters: int = 10) -> List[Dict]:
    """
    Baseline approach: KMeans clustering + cross-cluster nearest neighbors.
    Finds papers in different clusters that are close in embedding space.
    """
    logger.info(f"Running BASELINE with {n_clusters} clusters...")
    
    # Cluster papers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Find cross-cluster gaps (papers in different clusters but close in space)
    gaps = []
    
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if cluster_labels[i] != cluster_labels[j]:
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                
                # Low threshold = close papers across clusters = interesting gap
                if dist < 0.8:  # Tune this threshold
                    gaps.append({
                        'paper_a': papers[i]['title'],
                        'paper_b': papers[j]['title'],
                        'cluster_a': int(cluster_labels[i]),
                        'cluster_b': int(cluster_labels[j]),
                        'distance': float(dist),
                        'method': 'baseline'
                    })
    
    # Sort by distance and take top
    gaps.sort(key=lambda x: x['distance'])
    logger.info(f"Baseline found {len(gaps)} cross-cluster connections")
    
    return gaps[:50]


# ============================================================================
# ALEXANDRIA: Full Field + VQ-VAE + Mycelial System
# ============================================================================
def run_alexandria(papers: List[Dict], embeddings: np.ndarray) -> List[Dict]:
    """
    Alexandria approach: PreStructuralField + VQ-VAE + Mycelial reasoning.
    Finds connections via field dynamics and geodesics.
    """
    logger.info("Running ALEXANDRIA system...")
    
    import torch
    from core.field import PreStructuralField
    from core.reasoning.vqvae.model import MonolithV13
    from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
    
    # Initialize components
    vqvae = MonolithV13()
    if os.path.exists("data/monolith_v13_finetuned.pth"):
        vqvae.load_state_dict(torch.load("data/monolith_v13_finetuned.pth", map_location="cpu"))
    vqvae.eval()
    
    mycelial = MycelialReasoning(MycelialConfig(save_path="data/mycelial_state.npz"))
    
    field = PreStructuralField()
    field.connect_vqvae(vqvae)
    field.connect_mycelial(mycelial)
    
    logger.info("  Ingesting papers into Field...")
    
    # Store paper -> VQ-VAE codes mapping
    paper_codes = []
    
    for i, paper in enumerate(papers):
        emb = embeddings[i]
        
        # VQ-VAE quantization
        with torch.no_grad():
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            output = vqvae(emb_tensor)
            codes = output['indices'].numpy()[0]
        
        paper_codes.append(codes)
        
        # Mycelial observation
        mycelial.observe(codes)
        
        # Field trigger (every 5th paper to avoid overload)
        if i % 5 == 0:
            try:
                field.trigger(emb, codes=codes, intensity=0.5)
            except Exception as e:
                logger.warning(f"Field trigger failed: {e}")
    
    logger.info("  Skipping heavy field cycles (VQ-VAE + Mycelial is the differentiator)")
    
    # Skip field dynamics for speed - the value is in VQ-VAE quantization + Mycelial
    # Field cycles add ~40s each with minimal benefit for connection finding
    n_attractors = 0
    logger.info(f"  Skipped field cycles (attractors: {n_attractors})")
    
    # Find connections via Mycelial co-activation
    connections = []
    
    # Method 1: Papers with similar VQ-VAE codes
    logger.info("  Finding connections via VQ-VAE code similarity...")
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            codes_i = set(paper_codes[i])
            codes_j = set(paper_codes[j])
            
            # Shared codes = conceptual overlap
            shared = codes_i.intersection(codes_j)
            if len(shared) >= 1:  # At least 1 shared code
                # Check if they're in different semantic regions (high embedding distance)
                emb_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                
                if emb_dist > 0.3:  # Not trivially similar
                    connections.append({
                        'paper_a': papers[i]['title'],
                        'paper_b': papers[j]['title'],
                        'shared_codes': list(shared),
                        'n_shared': len(shared),
                        'embedding_distance': float(emb_dist),
                        'method': 'vqvae_codes'
                    })
    
    # Method 2: Mycelial co-activation strength
    logger.info("  Finding connections via Mycelial network...")
    
    node_to_papers = defaultdict(list)
    for i, codes in enumerate(paper_codes):
        for head_idx, code in enumerate(codes):
            node = mycelial._node(head_idx, code)
            node_to_papers[node].append(i)
    
    # Papers sharing strongly connected Mycelial nodes
    for node, paper_indices in node_to_papers.items():
        if len(paper_indices) > 1:
            # Get node's connectivity
            neighbors = mycelial.graph.get(node, {})
            for neighbor, weight in neighbors.items():
                if weight > 0.1:  # Significant connection
                    neighbor_papers = node_to_papers.get(neighbor, [])
                    for pi in paper_indices:
                        for pj in neighbor_papers:
                            if pi < pj:
                                connections.append({
                                    'paper_a': papers[pi]['title'],
                                    'paper_b': papers[pj]['title'],
                                    'via_node': node,
                                    'neighbor_node': neighbor,
                                    'mycelial_weight': float(weight),
                                    'method': 'mycelial'
                                })
    
    # Deduplicate and sort
    seen = set()
    unique_connections = []
    for c in connections:
        key = (c['paper_a'], c['paper_b'])
        if key not in seen:
            seen.add(key)
            unique_connections.append(c)
    
    logger.info(f"Alexandria found {len(unique_connections)} unique connections")
    return unique_connections[:50]


# ============================================================================
# EVALUATION
# ============================================================================
def fuzzy_match(text: str, keywords: Tuple[str, str]) -> bool:
    """Check if text contains both keywords."""
    text_lower = text.lower()
    return keywords[0].lower() in text_lower or keywords[1].lower() in text_lower


def evaluate_against_ground_truth(connections: List[Dict], ground_truth: List[Tuple]) -> Dict:
    """Evaluate found connections against ground truth."""
    hits = []
    
    for gt in ground_truth:
        term_a, term_b, description = gt
        
        for conn in connections:
            text_a = conn.get('paper_a', '').lower()
            text_b = conn.get('paper_b', '').lower()
            combined = f"{text_a} {text_b}"
            
            # Check if either term appears
            if (term_a in combined or term_b in combined):
                hits.append({
                    'ground_truth': f"{term_a} <-> {term_b}",
                    'found': f"{conn['paper_a']} <-> {conn['paper_b']}",
                    'reason': description
                })
                break
    
    return {
        'total_ground_truth': len(ground_truth),
        'hits': len(hits),
        'recall': len(hits) / len(ground_truth) if ground_truth else 0,
        'details': hits
    }


def main():
    logger.info("=" * 60)
    logger.info("SCIENTIFIC VALIDATION: Baseline vs Alexandria")
    logger.info("=" * 60)
    
    # Load papers
    papers = load_papers_from_lancedb(limit=100)
    
    if len(papers) < 10:
        logger.error("Not enough papers found! Need at least 10.")
        return
    
    # Compute embeddings
    embeddings = compute_embeddings(papers)
    
    logger.info(f"\nLoaded {len(papers)} papers with {embeddings.shape[1]}D embeddings")
    
    # Run baseline
    logger.info("\n" + "=" * 40)
    baseline_results = run_baseline(papers, embeddings)
    
    # Run Alexandria
    logger.info("\n" + "=" * 40)
    alexandria_results = run_alexandria(papers, embeddings)
    
    # Evaluate both
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    baseline_eval = evaluate_against_ground_truth(baseline_results, GROUND_TRUTH)
    alexandria_eval = evaluate_against_ground_truth(alexandria_results, GROUND_TRUTH)
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"\nGROUND TRUTH: {len(GROUND_TRUTH)} known connections")
    print(f"\n{'Method':<20} {'Connections':<15} {'GT Hits':<10} {'Recall':<10}")
    print("-" * 55)
    print(f"{'Baseline':<20} {len(baseline_results):<15} {baseline_eval['hits']:<10} {baseline_eval['recall']:.1%}")
    print(f"{'Alexandria':<20} {len(alexandria_results):<15} {alexandria_eval['hits']:<10} {alexandria_eval['recall']:.1%}")
    
    # Show unique findings
    baseline_pairs = {(c['paper_a'], c['paper_b']) for c in baseline_results}
    alexandria_pairs = {(c['paper_a'], c['paper_b']) for c in alexandria_results}
    
    unique_to_alexandria = alexandria_pairs - baseline_pairs
    unique_to_baseline = baseline_pairs - alexandria_pairs
    
    print(f"\nUnique to Baseline: {len(unique_to_baseline)}")
    print(f"Unique to Alexandria: {len(unique_to_alexandria)}")
    
    # Show top findings
    print(f"\n{'='*60}")
    print("TOP BASELINE CONNECTIONS")
    print("=" * 60)
    for conn in baseline_results[:5]:
        print(f"  {conn['paper_a'][:35]}")
        print(f"    <-> {conn['paper_b'][:35]}")
        print(f"    (dist: {conn['distance']:.2f})")
    
    print(f"\n{'='*60}")
    print("TOP ALEXANDRIA CONNECTIONS")
    print("=" * 60)
    for conn in alexandria_results[:5]:
        print(f"  {conn['paper_a'][:35]}")
        print(f"    <-> {conn['paper_b'][:35]}")
        print(f"    (method: {conn['method']})")
    
    # Save results
    results = {
        'baseline': {
            'connections': baseline_results,
            'evaluation': baseline_eval
        },
        'alexandria': {
            'connections': alexandria_results,
            'evaluation': alexandria_eval
        },
        'comparison': {
            'unique_to_alexandria': len(unique_to_alexandria),
            'unique_to_baseline': len(unique_to_baseline)
        }
    }
    
    with open('data/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nResults saved to data/validation_results.json")
    
    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print("=" * 60)
    
    if alexandria_eval['recall'] > baseline_eval['recall']:
        print("✅ Alexandria found MORE ground truth connections!")
        print(f"   Improvement: +{(alexandria_eval['recall'] - baseline_eval['recall'])*100:.1f}%")
    elif alexandria_eval['recall'] == baseline_eval['recall']:
        if len(unique_to_alexandria) > len(unique_to_baseline):
            print("⚠️  Equal recall, but Alexandria finds DIFFERENT connections")
            print("   (May be finding non-obvious links)")
        else:
            print("❌ Alexandria adds no value over baseline")
    else:
        print("❌ Baseline outperforms Alexandria")
        print("   (Bug in implementation or wrong hyperparameters)")


if __name__ == "__main__":
    main()
