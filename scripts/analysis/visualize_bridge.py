"""
Visualization Script for Alexandria's Bridge Agent and Semantic Space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import UMAP, handle if missing
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    logger.warning("UMAP not found. Semantic space visualization will be skipped.")
    HAS_UMAP = False

# Plot style configuration
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["font.size"] = 11

def load_data(data_dir: str = "./data"):
    """Load data from the specified directory."""
    data_path = Path(data_dir)
    
    data = {}
    
    # Load embeddings if available
    if (data_path / "embeddings.npy").exists():
        data["embeddings"] = np.load(data_path / "embeddings.npy")
    
    # Load codes if available
    if (data_path / "codes.npy").exists():
        data["codes"] = np.load(data_path / "codes.npy")
        
    # Load doc_ids if available
    if (data_path / "doc_ids.npy").exists():
        data["doc_ids"] = np.load(data_path / "doc_ids.npy")
        
    # Load domains if available
    if (data_path / "domains.npy").exists():
        data["domains"] = np.load(data_path / "domains.npy")
        
    # Load bridge candidates if available
    if (data_path / "bridge_candidates.parquet").exists():
        data["bridge_candidates"] = pd.read_parquet(data_path / "bridge_candidates.parquet")
    elif (data_path / "bridge_candidates.csv").exists():
        data["bridge_candidates"] = pd.read_csv(data_path / "bridge_candidates.csv")
        
    return data

def plot_semantic_space(embeddings, labels=None, label_type="doc_id", output_path=None):
    """Plot UMAP projection of embeddings."""
    if not HAS_UMAP:
        return
        
    logger.info("Generating UMAP projection...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    
    embeddings_2d = umap_model.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Create DataFrame for plotting
        df_map = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "label": labels
        })
        
        # If too many unique labels, just use scatter
        if len(np.unique(labels)) > 20 and label_type != "code":
            plt.scatter(df_map["x"], df_map["y"], s=8, alpha=0.7, c='blue')
            plt.title(f"Semantic Space (UMAP)")
        else:
            if label_type == "code":
                scatter = plt.scatter(
                    df_map["x"], 
                    df_map["y"], 
                    c=labels, 
                    s=8, 
                    alpha=0.7,
                    cmap='viridis'
                )
                plt.colorbar(scatter, label=f"VQ-VAE Code")
            else:
                unique_labels = df_map["label"].unique()
                for lbl in unique_labels:
                    subset = df_map[df_map["label"] == lbl]
                    plt.scatter(
                        subset["x"],
                        subset["y"],
                        s=10,
                        alpha=0.7,
                        label=str(lbl)[:20],
                    )
                plt.legend(loc="best", frameon=False, fontsize=8)
            
            plt.title(f"Semantic Space (UMAP) - Colored by {label_type}")
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=8, alpha=0.7)
        plt.title("Semantic Space (UMAP)")

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved semantic space plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_code_usage(codes, output_path=None):
    """Plot histogram of VQ-VAE code usage."""
    num_heads = codes.shape[1]
    codebook_size = int(codes.max()) + 1
    
    fig, axes = plt.subplots(num_heads, 1, figsize=(10, 3 * num_heads), sharex=True)
    
    if num_heads == 1:
        axes = [axes]
        
    for h in range(num_heads):
        ax = axes[h]
        head_codes = codes[:, h]
        counts = np.bincount(head_codes, minlength=codebook_size)
        
        ax.bar(np.arange(codebook_size), counts, alpha=0.7)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Code Usage - Head {h}")
        # Only show integer ticks if codebook is small
        if codebook_size <= 20:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
    axes[-1].set_xlabel("Code Index")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved code usage plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_code_coactivation(codes, head_a=0, head_b=1, output_path=None):
    """Plot co-activation matrix between two heads."""
    if codes.shape[1] <= max(head_a, head_b):
        logger.warning(f"Cannot plot co-activation: codes shape {codes.shape} too small for heads {head_a}, {head_b}")
        return

    codes_a = codes[:, head_a]
    codes_b = codes[:, head_b]
    
    codebook_size = max(int(codes.max()) + 1, 256) # Default to at least 256
    
    co_matrix = np.zeros((codebook_size, codebook_size), dtype=int)
    for ca, cb in zip(codes_a, codes_b):
        if ca < codebook_size and cb < codebook_size:
            co_matrix[ca, cb] += 1
            
    plt.figure(figsize=(8, 7))
    plt.imshow(co_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Co-occurrence Frequency")
    plt.title(f"Code Co-activation (Head {head_a} vs Head {head_b})")
    plt.xlabel(f"Code Head {head_b}")
    plt.ylabel(f"Code Head {head_a}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved co-activation plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_bridge_candidates(df, output_path=None):
    """Plot Bridge Agent candidate analysis."""
    if df is None or df.empty:
        logger.warning("No bridge candidates data to plot.")
        return
        
    required_cols = ["sim_bridge", "novelty", "final_score"]
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Bridge candidates dataframe missing required columns: {required_cols}")
        return

    # 1. Scatter: Similarity vs Novelty
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["sim_bridge"],
        df["novelty"],
        s=40,
        alpha=0.7,
        c=df["final_score"],
        cmap="viridis"
    )
    plt.colorbar(label="Final Score")
    
    plt.xlabel("Similarity to Bridge Vector")
    plt.ylabel("Novelty (vs Memory)")
    plt.title("Bridge Agent: Similarity vs Novelty Trade-off")
    
    # Highlight top candidates
    top_n = min(5, len(df))
    top = df.sort_values("final_score", ascending=False).head(top_n)
    
    plt.scatter(
        top["sim_bridge"],
        top["novelty"],
        s=100,
        facecolors='none',
        edgecolors='red',
        linewidth=2,
        label="Top Candidates"
    )
    
    for _, row in top.iterrows():
        label = row.get("doc_id", "Unknown")
        # Shorten if too long
        if isinstance(label, str) and len(label) > 15:
            label = label[:12] + "..."
            
        plt.text(
            row["sim_bridge"],
            row["novelty"],
            str(label),
            fontsize=9,
            ha="left",
            va="bottom",
            fontweight='bold'
        )
        
    plt.legend(loc="best")
    plt.tight_layout()
    
    if output_path:
        scatter_path = str(output_path).replace(".png", "_scatter.png")
        plt.savefig(scatter_path)
        logger.info(f"Saved bridge scatter plot to {scatter_path}")
    else:
        plt.show()
    plt.close()
    
    # 2. Score Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["final_score"], bins=30, alpha=0.7, color='teal')
    plt.xlabel("Final Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Bridge Candidate Scores")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    
    if output_path:
        hist_path = str(output_path).replace(".png", "_hist.png")
        plt.savefig(hist_path)
        logger.info(f"Saved bridge histogram to {hist_path}")
    else:
        plt.show()
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Alexandria's Internal State")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--output-dir", type=str, default="./reports", help="Directory to save plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    data = load_data(args.data_dir)
    
    # 1. Semantic Space
    if "embeddings" in data:
        labels = None
        label_type = "doc_id"
        
        if "domains" in data:
            labels = data["domains"]
            label_type = "domain"
        elif "codes" in data:
            # Use head 0 codes as labels
            labels = data["codes"][:, 0]
            label_type = "code"
            
        plot_semantic_space(
            data["embeddings"], 
            labels=labels, 
            label_type=label_type,
            output_path=output_dir / "semantic_space.png"
        )
    else:
        logger.info("Skipping semantic space plot (no embeddings found)")
        
    # 2. Code Usage
    if "codes" in data:
        plot_code_usage(
            data["codes"], 
            output_path=output_dir / "code_usage.png"
        )
        
        if data["codes"].shape[1] >= 2:
            plot_code_coactivation(
                data["codes"], 
                output_path=output_dir / "code_coactivation.png"
            )
    else:
        logger.info("Skipping code usage plots (no codes found)")
        
    # 3. Bridge Candidates
    if "bridge_candidates" in data:
        plot_bridge_candidates(
            data["bridge_candidates"],
            output_path=output_dir / "bridge_analysis.png"
        )
    else:
        logger.info("Skipping bridge candidate plots (no candidate data found)")
        
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
