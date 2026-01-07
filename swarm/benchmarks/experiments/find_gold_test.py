"""
Test: Finding "Gold" in Similar Documents

Demonstrates how find_outliers() can identify unique content 
even when all documents are semantically similar.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

print("=" * 60)
print("FINDING GOLD IN SIMILAR DOCUMENTS")
print("=" * 60)

from core.topology.topology_engine import create_topology_engine
topology = create_topology_engine()

# Simulate a specialized database: AI papers
papers = [
    "Neural networks use backpropagation for training",
    "Deep learning models learn hierarchical representations",
    "Convolutional networks are effective for image tasks",
    "Recurrent networks process sequential data",
    "Transformer architecture uses attention mechanisms",
    "LSTM networks handle long-term dependencies",
    "Batch normalization improves training stability",
    # THE GOLD - something slightly different
    "Quantum neural networks leverage quantum superposition for exponential speedup",
    "Neuromorphic computing mimics biological neural systems in hardware",
]

paper_labels = [
    "Standard: Backprop",
    "Standard: Deep Learning",
    "Standard: CNNs",
    "Standard: RNNs",
    "Standard: Transformers",
    "Standard: LSTMs",
    "Standard: BatchNorm",
    "★ GOLD: Quantum Neural Networks",
    "★ GOLD: Neuromorphic Computing",
]

print("\n[1] Generating embeddings for all papers...")
embeddings = topology.encode(papers)
print(f"Generated {len(embeddings)} embeddings")

# Show that most papers are similar to each other
print("\n[2] Similarity matrix (all papers are similar):")
print("     " + "  ".join([f"{i}" for i in range(len(papers))]))
for i in range(len(papers)):
    sims = []
    for j in range(len(papers)):
        sim = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
        )
        sims.append(f"{sim:.2f}")
    print(f"  {i}: " + " ".join(sims))

# Find outliers - the unique papers
print("\n[3] Finding OUTLIERS (unique content)...")
outliers = topology.find_outliers(embeddings, paper_labels, top_k=3)

print("\nTop 3 most unique papers (potential GOLD):")
for i, (emb, distance, label) in enumerate(outliers):
    print(f"  {i+1}. Distance from centroid: {distance:.4f}")
    print(f"     {label}")

# Show that the gold was found
print("\n" + "=" * 60)
gold_found = any("GOLD" in label for _, _, label in outliers[:2])
if gold_found:
    print("✅ SUCCESS: The system identified the unique papers!")
else:
    print("❌ FAILED: Gold not in top outliers")
print("=" * 60)
