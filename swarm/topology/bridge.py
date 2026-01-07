"""
swarm/topology/bridge.py

ActiveBridgeBuilder: Constructs hypothetical conceptual bridges between concepts.

Instead of passively querying the Mycelial Graph, this module actively proposes
and validates potential bridge concepts that could connect two distant points
in the semantic space.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class BridgeCandidate:
    """A proposed bridge between two concepts."""
    concept: Any  # The bridge concept (embedding or ID)
    confidence: float
    method: str  # 'analogy', 'interpolation', 'relational'
    path: List[Any] = field(default_factory=list)
    validation_score: float = 0.0

class ActiveBridgeBuilder:
    """
    Constructs hypothetical bridges between concepts.
    
    Methods:
    - Structural Analogy: A:B :: C:? => find ? (D)
    - Vector Interpolation: Linearly interpolate and find nearest concepts
    - Relational Transposition: Transfer relationships from known pairs
    """
    
    def __init__(
        self,
        topology_engine=None,
        mycelial_graph=None,
        validation_threshold: float = 0.6
    ):
        self.topology = topology_engine
        self.graph = mycelial_graph
        self.validation_threshold = validation_threshold
        
    def propose_bridge(
        self,
        concept_a: np.ndarray,
        concept_b: np.ndarray,
        method: str = 'interpolation',
        num_candidates: int = 3
    ) -> List[BridgeCandidate]:
        """
        Proposes bridge candidates between two concepts.
        
        Args:
            concept_a: Start embedding
            concept_b: Target embedding
            method: 'interpolation' or 'analogy'
            num_candidates: Number of candidates to return
            
        Returns:
            List of BridgeCandidate sorted by confidence
        """
        if method == 'interpolation':
            return self._propose_by_interpolation(concept_a, concept_b, num_candidates)
        elif method == 'analogy':
            return self._propose_by_analogy(concept_a, concept_b, num_candidates)
        else:
            # Default to interpolation
            return self._propose_by_interpolation(concept_a, concept_b, num_candidates)
    
    def _propose_by_interpolation(
        self,
        A: np.ndarray,
        B: np.ndarray,
        num_candidates: int = 3
    ) -> List[BridgeCandidate]:
        """
        Proposes bridges via linear interpolation in embedding space.
        
        Steps:
        1. Create interpolated points between A and B
        2. For each point, find nearest real concepts
        3. Validate and score
        """
        candidates = []
        
        # Create interpolation points (excluding endpoints)
        steps = 5
        for i in range(1, steps):
            t = i / steps
            # Spherical linear interpolation for normalized vectors
            interpolated = self._slerp(A, B, t)
            
            # Find nearest concept to this interpolated point
            if self.topology:
                nearest = self._find_nearest_concept(interpolated)
                if nearest is not None:
                    # Calculate confidence based on position quality
                    sim_a = self._cosine_similarity(nearest, A)
                    sim_b = self._cosine_similarity(nearest, B)
                    
                    # Good bridge should be somewhat similar to both
                    confidence = min(sim_a, sim_b) * (sim_a + sim_b) / 2
                    
                    candidate = BridgeCandidate(
                        concept=nearest,
                        confidence=float(confidence),
                        method='interpolation',
                        path=[A, nearest, B]
                    )
                    candidates.append(candidate)
            else:
                # Fallback: use the interpolated point itself as candidate
                confidence = 0.5
                candidate = BridgeCandidate(
                    concept=interpolated,
                    confidence=confidence,
                    method='interpolation',
                    path=[A, interpolated, B]
                )
                candidates.append(candidate)
        
        # Sort by confidence and return top N
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[:num_candidates]
    
    def _propose_by_analogy(
        self,
        A: np.ndarray,
        B: np.ndarray,
        num_candidates: int = 3
    ) -> List[BridgeCandidate]:
        """
        Proposes bridges via structural analogy.
        
        If we know A relates to X by relation R,
        find Y that relates to B by similar relation.
        The path A -> X -> Y -> B forms a bridge.
        
        Requires a populated Mycelial Graph.
        """
        candidates = []
        
        if not self.graph:
            # Cannot do analogy without graph
            return candidates
        
        # Get neighbors of A in the graph
        neighbors_a = self.graph.get_neighbors(A) if hasattr(self.graph, 'get_neighbors') else []
        
        for neighbor in neighbors_a[:5]:  # Top 5 neighbors
            # Get the relation vector (A -> neighbor)
            relation = neighbor - A  # Simplified: vector difference
            
            # Apply same relation from B
            potential_bridge = B + relation
            
            # Normalize
            norm = np.linalg.norm(potential_bridge)
            if norm > 0:
                potential_bridge = potential_bridge / norm
            
            # Find nearest real concept
            nearest = self._find_nearest_concept(potential_bridge)
            if nearest is not None:
                confidence = self._cosine_similarity(nearest, potential_bridge)
                candidate = BridgeCandidate(
                    concept=nearest,
                    confidence=float(confidence),
                    method='analogy',
                    path=[A, neighbor, nearest, B]
                )
                candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[:num_candidates]
    
    def validate_bridge(self, candidate: BridgeCandidate) -> Tuple[bool, str]:
        """
        Validates a bridge candidate before adding to the graph.
        
        Criteria:
        1. Semantic coherence: Path makes sense
        2. Not redundant: Doesn't duplicate existing paths
        3. Useful: Actually reduces distance between concepts
        """
        path = candidate.path
        if len(path) < 3:
            return False, "Path too short"
        
        A, bridge, B = path[0], path[len(path)//2], path[-1]
        
        # 1. Semantic coherence
        sim_a_bridge = self._cosine_similarity(A, bridge)
        sim_bridge_b = self._cosine_similarity(bridge, B)
        
        if sim_a_bridge < 0.3 or sim_bridge_b < 0.3:
            return False, f"Low semantic coherence: {sim_a_bridge:.2f}, {sim_bridge_b:.2f}"
        
        # 2. Check if bridge actually helps
        direct_sim = self._cosine_similarity(A, B)
        bridged_path_quality = (sim_a_bridge + sim_bridge_b) / 2
        
        if bridged_path_quality <= direct_sim:
            return False, "Bridge doesn't improve path"
        
        # 3. Update validation score
        candidate.validation_score = bridged_path_quality
        
        return True, "Bridge valid"
    
    def _slerp(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation."""
        A = A / (np.linalg.norm(A) + 1e-9)
        B = B / (np.linalg.norm(B) + 1e-9)
        
        dot = np.clip(np.dot(A, B), -1.0, 1.0)
        theta = np.arccos(dot)
        
        if theta < 1e-6:
            return A
        
        sin_theta = np.sin(theta)
        result = (np.sin((1 - t) * theta) / sin_theta) * A + (np.sin(t * theta) / sin_theta) * B
        return result / (np.linalg.norm(result) + 1e-9)
    
    def _cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(A)
        norm_b = np.linalg.norm(B)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(A, B) / (norm_a * norm_b))
    
    def _find_nearest_concept(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Find the nearest real concept to a point in embedding space."""
        if self.topology and hasattr(self.topology, 'find_nearest'):
            return self.topology.find_nearest(point)
        # If no topology engine, return None
        return None
