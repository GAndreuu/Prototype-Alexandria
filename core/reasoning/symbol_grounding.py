"""
Symbol Grounding Module
=======================

Bridges the gap between symbolic (text) and subsymbolic (neural/graph) representations.

Functionality:
1. Embeds text using TopologyEngine (SimCSE/SentenceTransformer).
2. Quantizes embedding using MycelialVQVAE.
3. Returns discrete Mycelial Nodes (head, code).

Usage:
    grounder = SymbolGrounder()
    nodes = grounder.ground("machine learning")
    # nodes = [(0, 12), (1, 55), (2, 128), (3, 9)]
"""

import logging
import torch
import numpy as np
from typing import List, Tuple, Optional, Any

# Conditional imports to avoid circular deps or hard failures
try:
    from core.topology.topology_engine import TopologyEngine
except ImportError:
    TopologyEngine = None

try:
    from core.reasoning.mycelial_reasoning import MycelialVQVAE, MycelialReasoning, Node
except ImportError:
    MycelialVQVAE = None
    MycelialReasoning = None
    Node = Tuple[int, int]

logger = logging.getLogger(__name__)

class SymbolGrounder:
    """
    Grounds textual concepts into Mycelial Graph nodes.
    """
    
    def __init__(
        self, 
        topology_engine: Optional[Any] = None,
        vqvae_wrapper: Optional[Any] = None
    ):
        self.topology = topology_engine
        self.vqvae = vqvae_wrapper
        
        # Lazy initialization
        if self.topology is None and TopologyEngine:
            try:
                self.topology = TopologyEngine()
                logger.info("SymbolGrounder: TopologyEngine initialized.")
            except Exception as e:
                logger.warning(f"SymbolGrounder: Failed to init TopologyEngine: {e}")
                
        if self.vqvae is None and MycelialVQVAE:
            try:
                self.vqvae = MycelialVQVAE.load_default()
                logger.info("SymbolGrounder: MycelialVQVAE initialized.")
            except Exception as e:
                logger.warning(f"SymbolGrounder: Failed to init MycelialVQVAE: {e}")

    def ground(self, text: str) -> List[Node]:
        """
        Converts text string to a list of (head, code) tuples.
        """
        if not text:
            return []
            
        if not self.topology:
            logger.warning("SymbolGrounder: No TopologyEngine, cannot embed text.")
            return []
            
        if not self.vqvae:
            logger.warning("SymbolGrounder: No VQ-VAE, cannot quantize.")
            return []
            
        try:
            # 1. Embed text -> Vector
            # TopologyEngine.encode returns list, we take first
            embeddings = self.topology.encode([text])
            if embeddings is None or len(embeddings) == 0:
                logger.warning(f"SymbolGrounder: Embedding failed for '{text}'")
                return []
                
            vector = embeddings[0]
            
            # 2. Vector -> Tensor
            t_vector = torch.tensor(vector, dtype=torch.float32)
            
            # 3. Quantize via VQ-VAE
            # MycelialVQVAE.encode expects tensor, returns indices tensor
            # Ensure device
            if hasattr(self.vqvae.vqvae, 'device'):
                t_vector = t_vector.to(self.vqvae.vqvae.device)
                
            indices = self.vqvae.encode(t_vector)
            
            # 4. Convert to Nodes
            nodes = []
            indices_flat = indices.flatten().cpu().numpy()
            for h, code in enumerate(indices_flat):
                nodes.append((int(h), int(code)))
                
            return nodes
            
        except Exception as e:
            logger.error(f"SymbolGrounder: Grounding failed for '{text}': {e}")
            return []

    def ground_gap(self, gap_description: str) -> List[Node]:
        """Wrapper for explicit usage with gaps."""
        return self.ground(gap_description)
