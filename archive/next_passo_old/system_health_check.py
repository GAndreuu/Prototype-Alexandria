"""
Alexandria System Health Check
Complete diagnostic of codebook distribution, system health, and data quality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sentence_transformers import SentenceTransformer
from core.reasoning.mycelial_reasoning import MycelialVQVAE
from core.memory.storage import LanceDBStorage
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_codebook_distribution(num_samples=5000):
    """Analyze actual codebook usage distribution."""
    
    logger.info("="*80)
    logger.info("ALEXANDRIA SYSTEM HEALTH CHECK")
    logger.info("="*80)
    
    # Load models
    logger.info("\n📊 Loading models...")
    wrapper = MycelialVQVAE.load_default()
    storage = LanceDBStorage()
    
    # Sample embeddings
    logger.info(f"\n📦 Sampling {num_samples} embeddings from database...")
    table = storage.table
    df = table.to_pandas()
    sample_df = df.sample(n=min(num_samples, len(df)))
    
    # Encode all samples
    logger.info(f"\n🔢 Encoding {len(sample_df)} embeddings...")
    all_indices = []
    
    for idx, row in sample_df.iterrows():
        vector = row['vector']
        indices = wrapper.encode(vector)
        all_indices.append(indices)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"   Encoded {idx + 1}/{len(sample_df)}")
    
    all_indices = np.array(all_indices)  # [N, 4]
    
    # Analyze per-head distribution
    logger.info("\n" + "="*80)
    logger.info("CODEBOOK USAGE ANALYSIS")
    logger.info("="*80)
    
    for head in range(4):
        codes = all_indices[:, head]
        unique, counts = np.unique(codes, return_counts=True)
        
        logger.info(f"\n🔷 HEAD {head}")
        logger.info(f"   Unique codes used: {len(unique)}/256 ({len(unique)/256*100:.1f}%)")
        logger.info(f"   Most frequent code: {unique[counts.argmax()]} ({counts.max()} occurrences)")
        logger.info(f"   Least frequent code: {unique[counts.argmin()]} ({counts.min()} occurrences)")
        logger.info(f"   Mean frequency: {counts.mean():.1f}")
        logger.info(f"   Std frequency: {counts.std():.1f}")
        
        # Top 10 codes
        top_10_idx = counts.argsort()[-10:][::-1]
        logger.info(f"\n   Top 10 codes:")
        for i, idx in enumerate(top_10_idx, 1):
            logger.info(f"      {i}. Code {unique[idx]}: {counts[idx]} times ({counts[idx]/len(codes)*100:.2f}%)")
    
    # Overall statistics
    logger.info("\n" + "="*80)
    logger.info("OVERALL STATISTICS")
    logger.info("="*80)
    
    # Total unique combinations
    unique_combinations = len(np.unique(all_indices, axis=0))
    logger.info(f"\n🎯 Unique code combinations: {unique_combinations}/{len(all_indices)}")
    logger.info(f"   Diversity ratio: {unique_combinations/len(all_indices)*100:.2f}%")
    
    # Power-law analysis
    logger.info("\n📈 POWER-LAW ANALYSIS")
    all_codes_flat = all_indices.flatten()
    code_counts = Counter(all_codes_flat)
    
    # Sort by frequency
    sorted_counts = sorted(code_counts.values(), reverse=True)
    
    # Fit power law  (y = x^-α)
    ranks = np.arange(1, len(sorted_counts) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_counts)
    
    # Linear fit in log-log space
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    alpha = -coeffs[0]
    
    logger.info(f"   Power-law exponent (α): {alpha:.3f}")
    
    if 1.2 <= alpha <= 2.0:
        logger.info(f"   ✅ HEALTHY (Zipf-like distribution)")
    elif alpha < 1.2:
        logger.info(f"   ⚠️  Too flat (uniform-like)")
    else:
        logger.info(f"   ⚠️  Too steep (concentrated)")
    
    # R² of fit
    predicted = np.exp(coeffs[0] * log_ranks + coeffs[1])
    ss_res = np.sum((sorted_counts - predicted) ** 2)
    ss_tot = np.sum((sorted_counts - np.mean(sorted_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    logger.info(f"   R² of power-law fit: {r_squared:.3f}")
    
    # System health
    logger.info("\n" + "="*80)
    logger.info("SYSTEM HEALTH")
    logger.info("="*80)
    
    # Database
    total_records = len(df)
    logger.info(f"\n💾 DATABASE:")
    logger.info(f"   Total records: {total_records:,}")
    logger.info(f"   Embedding dimension: {len(df.iloc[0]['vector'])}")
    
    # Mycelial network
    stats = wrapper.get_network_stats()
    logger.info(f"\n🍄 MYCELIAL NETWORK:")
    logger.info(f"   Observations: {stats.get('total_observations', 0):,}")
    logger.info(f"   Active connections: {stats.get('active_connections', 0):,}")
    logger.info(f"   Density: {stats.get('density', 0):.6f}")
    
    expected_connections = stats.get('total_observations', 0) * 0.1
    actual = stats.get('active_connections', 0)
    
    if actual >= expected_connections * 0.5:
        logger.info(f"   ✅ Connection density is healthy")
    else:
        logger.info(f"   ⚠️  Few connections (expected ~{expected_connections:.0f}, got {actual})")
    
    # Codebooks
    codebooks = wrapper.get_codebooks()  # [4, 256, 128]
    logger.info(f"\n🧬 CODEBOOKS:")
    logger.info(f"   Shape: {codebooks.shape}")
    
    # Check diversity of codebooks
    for h in range(4):
        cb = codebooks[h]
        # Pairwise distances
        norms = np.linalg.norm(cb, axis=1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        
        logger.info(f"\n   Head {h}:")
        logger.info(f"      Mean norm: {mean_norm:.3f}")
        logger.info(f"      Std norm: {std_norm:.3f}")
        
        # Sample pairwise distance
        sample_idx = np.random.choice(256, size=100, replace=False)
        sample_codes = cb[sample_idx]
        dists = np.linalg.norm(sample_codes[:, None] - sample_codes[None, :], axis=2)
        mean_dist = dists[np.triu_indices(100, k=1)].mean()
        
        logger.info(f"      Avg pairwise distance: {mean_dist:.3f}")
        
        if mean_dist > 1.0:
            logger.info(f"      ✅ Codes are well-separated")
        else:
            logger.info(f"      ⚠️  Codes might be too similar")
    
    # Generate plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOTS")
    logger.info("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Alexandria System Health Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Code usage per head
    for head in range(4):
        codes = all_indices[:, head]
        unique, counts = np.unique(codes, return_counts=True)
        
        ax = axes[0, 0] if head < 2 else axes[1, 0]
        offset = 0 if head % 2 == 0 else 128
        
        ax.bar(unique + offset, counts, width=1, alpha=0.7, label=f'Head {head}')
        ax.set_xlabel('Code Index')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Code Usage Distribution (Heads {0 if head < 2 else 2}-{1 if head < 2 else 3})')
        ax.legend()
    
    # Plot 2: Power-law
    ax = axes[0, 1]
    ax.loglog(ranks, sorted_counts, 'o', alpha=0.5, label='Data')
    ax.loglog(ranks, predicted, 'r-', label=f'Fit (α={alpha:.2f})')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Power-Law Distribution\nR²={r_squared:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Unique combinations
    ax = axes[0, 2]
    ax.bar(['Used', 'Unused'], 
           [unique_combinations, len(all_indices) - unique_combinations],
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Code Combination Diversity')
    
    # Plot 4: Code frequency heatmap
    ax = axes[1, 1]
    usage_matrix = np.zeros((4, 256))
    for head in range(4):
        codes = all_indices[:, head]
        unique, counts = np.unique(codes, return_counts=True)
        usage_matrix[head, unique] = counts
    
    im = ax.imshow(usage_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Code Index')
    ax.set_ylabel('Head')
    ax.set_title('Code Usage Heatmap')
    plt.colorbar(im, ax=ax, label='Frequency')
    
    # Plot 5: Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Database:
  • Records: {total_records:,}
  • Samples analyzed: {len(all_indices):,}

Codebook Usage:
  • Total unique codes: {len(np.unique(all_indices.flatten()))}
  • Avg usage per head: {usage_matrix.sum(axis=1).mean()/256*100:.1f}%

Power-Law:
  • Exponent (α): {alpha:.3f}
  • R²: {r_squared:.3f}
  • {'✅ Healthy' if 1.2 <= alpha <= 2.0 else '⚠️ Check'}

Mycelial:
  • Observations: {stats.get('total_observations', 0):,}
  • Connections: {stats.get('active_connections', 0):,}
  • Density: {stats.get('density', 0):.4%}
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = Path('system_health_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✅ Dashboard saved: {output_path.absolute()}")
    
    logger.info("\n" + "="*80)
    logger.info("HEALTH CHECK COMPLETE")
    logger.info("="*80)
    
    # Final verdict
    logger.info("\n🎯 FINAL VERDICT:")
    
    issues = []
    if alpha < 1.2 or alpha > 2.0:
        issues.append(f"Power-law exponent out of range ({alpha:.2f})")
    if r_squared < 0.8:
        issues.append(f"Poor power-law fit (R²={r_squared:.2f})")
    if actual < expected_connections * 0.3:
        issues.append(f"Very few Mycelial connections ({actual} << {expected_connections:.0f})")
    
    if not issues:
        logger.info("   ✅ SYSTEM IS HEALTHY")
    else:
        logger.info("   ⚠️  ISSUES DETECTED:")
        for issue in issues:
            logger.info(f"      - {issue}")
    
    logger.info(f"\nView dashboard: {output_path.absolute()}")
    
    return {
        'codebook_usage': usage_matrix,
        'power_law_alpha': alpha,
        'r_squared': r_squared,
        'unique_combinations': unique_combinations,
        'stats': stats
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    args = parser.parse_args()
    
    results = analyze_codebook_distribution(args.samples)
