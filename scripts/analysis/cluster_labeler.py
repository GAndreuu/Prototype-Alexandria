"""
Cluster Labeler - Extract meaningful labels from indexed papers
================================================================
Maps cluster IDs to human-readable labels by analyzing paper titles/content.

Usage:
    python scripts/cluster_labeler.py --output data/cluster_labels.json
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClusterLabeler")


class ClusterLabeler:
    """
    Extracts meaningful labels for topology clusters from indexed papers.
    
    Process:
    1. Load all papers from LanceDB
    2. Assign each paper to a cluster using TopologyEngine
    3. Extract keywords/titles from each cluster
    4. Generate representative label for each cluster
    """
    
    def __init__(self):
        self.topology = None
        self.storage = None
        self.cluster_papers: Dict[int, List[dict]] = defaultdict(list)
        self.cluster_labels: Dict[int, str] = {}
    
    def initialize(self):
        """Load TopologyEngine and LanceDB storage."""
        logger.info("Initializing ClusterLabeler...")
        
        # Load Topology
        from core.topology.topology_engine import TopologyEngine
        self.topology = TopologyEngine()
        
        if os.path.exists("data/topology.json"):
            self.topology.load_topology("data/topology.json")
            logger.info(f"Loaded topology: {self.topology.n_clusters} clusters")
        else:
            logger.error("No topology found! Train topology first.")
            return False
        
        # Load LanceDB
        from core.memory.storage import LanceDBStorage
        self.storage = LanceDBStorage()
        logger.info(f"LanceDB loaded: {self.storage.count()} documents")
        
        return True
    
    def extract_labels(self, sample_size: int = 10000) -> Dict[int, str]:
        """
        Extract labels for all clusters using training embeddings + LanceDB content.
        """
        logger.info(f"Extracting labels using training embeddings...")
        
        # Load training embeddings
        embeddings_path = "data/training_embeddings.npy"
        if not os.path.exists(embeddings_path):
            logger.error("No training_embeddings.npy found!")
            return {}
        
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        
        # Use topology to get cluster assignments (via centroids, not predict to avoid sklearn version issues)
        if not self.topology.is_trained or self.topology.kmeans is None:
            logger.error("Topology not trained!")
            return {}
        
        # Get cluster centroids
        centroids = self.topology.kmeans.cluster_centers_
        
        # Assign embeddings to clusters using L2 distance (faster than calling predict)
        logger.info("Assigning embeddings to clusters...")
        sample_embeddings = embeddings[:sample_size]
        
        # Compute distances to all centroids in batches
        batch_size = 1000
        cluster_assignments = []
        for i in range(0, len(sample_embeddings), batch_size):
            batch = sample_embeddings[i:i+batch_size]
            # L2 distance to each centroid
            distances = np.linalg.norm(batch[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            batch_assignments = np.argmin(distances, axis=1)
            cluster_assignments.extend(batch_assignments)
        
        cluster_assignments = np.array(cluster_assignments)  # Convert to numpy array
        
        # Count documents per cluster
        cluster_counts = Counter(cluster_assignments)
        logger.info(f"Documents distributed across {len(cluster_counts)} clusters")
        
        # Now get sample documents from LanceDB for each cluster
        logger.info("Retrieving sample documents from LanceDB...")
        
        for cluster_id in range(self.topology.n_clusters):
            count = cluster_counts.get(cluster_id, 0)
            if count == 0:
                self.cluster_labels[cluster_id] = f"empty_cluster_{cluster_id}"
                continue
            
            # Get embeddings for this cluster
            cluster_indices = np.where(cluster_assignments == cluster_id)[0][:10]
            
            # Query LanceDB with these embeddings
            papers = []
            for idx in cluster_indices[:5]:  # Sample 5 papers per cluster
                try:
                    vec = embeddings[idx].astype(np.float32).tolist()
                    results = self.storage.search(vec, limit=1)
                    if results:
                        doc = results[0]
                        papers.append({
                            'title': self._extract_title(doc),
                            'content': doc.get('content', '')[:300],
                            'source': doc.get('source', '')
                        })
                except:
                    continue
            
            self.cluster_papers[cluster_id] = papers
            
            # Generate label from papers
            if papers:
                self.cluster_labels[cluster_id] = self._generate_label(cluster_id, papers)
            else:
                self.cluster_labels[cluster_id] = f"cluster_{cluster_id}"
        
        logger.info(f"Generated labels for {len(self.cluster_labels)} clusters")
        return self.cluster_labels
        
        # Generate labels for each cluster
        logger.info("Generating cluster labels...")
        for cluster_id, papers in self.cluster_papers.items():
            label = self._generate_label(cluster_id, papers)
            self.cluster_labels[cluster_id] = label
        
        return self.cluster_labels
    
    def _extract_title(self, doc: dict) -> str:
        """Extract title from document."""
        # Try metadata first
        meta = doc.get('metadata', {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except:
                meta = {}
        
        # Check various title fields
        title = meta.get('title') or meta.get('name') or meta.get('filename', '')
        
        if not title:
            # Try to extract from source (often contains filename)
            source = doc.get('source', '')
            if source:
                # Extract filename without extension
                filename = os.path.basename(source)
                title = os.path.splitext(filename)[0]
                # Clean up underscores and dashes
                title = title.replace('_', ' ').replace('-', ' ')
        
        if not title:
            # Fall back to first line of content
            content = doc.get('content', '')
            title = content.split('\n')[0][:100] if content else "Untitled"
        
        return title
    
    def _generate_label(self, cluster_id: int, papers: List[dict]) -> str:
        """Generate a representative label for a cluster."""
        if not papers:
            return f"cluster_{cluster_id}"
        
        # Extract all words from titles
        all_words = []
        for paper in papers[:50]:  # Use up to 50 papers
            title = paper.get('title', '')
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            all_words.extend(words)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
            'has', 'have', 'been', 'were', 'will', 'can', 'may', 'could', 'would',
            'should', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'over', 'using', 'based',
            'new', 'novel', 'approach', 'method', 'model', 'models', 'methods',
            'arxiv', 'pdf', 'paper', 'study', 'analysis', 'research'
        }
        
        filtered_words = [w for w in all_words if w not in stop_words]
        
        # Get most common words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(3)
        
        if top_words:
            # Create label from top words
            label = ' + '.join([w[0] for w in top_words])
            return label
        else:
            return f"cluster_{cluster_id}"
    
    def get_papers_for_cluster(self, cluster_id: int, limit: int = 5) -> List[dict]:
        """Get representative papers for a cluster."""
        papers = self.cluster_papers.get(cluster_id, [])
        return papers[:limit]
    
    def save_labels(self, output_path: str = "data/cluster_labels.json"):
        """Save cluster labels to file."""
        output = {
            "n_clusters": len(self.cluster_labels),
            "labels": {str(k): v for k, v in self.cluster_labels.items()},
            "sample_papers": {
                str(k): [p['title'] for p in v[:3]] 
                for k, v in self.cluster_papers.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.cluster_labels)} cluster labels to {output_path}")
    
    def print_summary(self, limit: int = 20):
        """Print summary of top clusters."""
        print("\n" + "=" * 60)
        print("CLUSTER LABELS SUMMARY")
        print("=" * 60)
        
        # Sort by number of papers
        sorted_clusters = sorted(
            self.cluster_papers.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for cluster_id, papers in sorted_clusters[:limit]:
            label = self.cluster_labels.get(cluster_id, "unknown")
            print(f"\nCluster {cluster_id}: {label}")
            print(f"  Papers: {len(papers)}")
            if papers:
                print(f"  Sample: {papers[0]['title'][:60]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract cluster labels from papers")
    parser.add_argument("--output", default="data/cluster_labels.json", help="Output file")
    parser.add_argument("--sample-size", type=int, default=10000, help="Max documents to process")
    args = parser.parse_args()
    
    labeler = ClusterLabeler()
    
    if not labeler.initialize():
        return
    
    labels = labeler.extract_labels(sample_size=args.sample_size)
    labeler.save_labels(args.output)
    labeler.print_summary()


if __name__ == "__main__":
    main()
