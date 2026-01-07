"""
swarm/topology/analyzer.py

Topological Analyzer.
Consolidated from v3/topology/analyzer.py.

Analyzes the topological structure of navigation trajectories and
semantic space to detect:
- Curvature and deformation
- Collapse risk
- Conceptual density
- Phase transitions
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..core import TopologicalState

@dataclass
class TopologyMetrics:
    """Aggregated topological metrics for a trajectory."""
    mean_curvature: float
    max_curvature: float
    total_length: float
    efficiency: float  # Direct distance / path length
    collapse_events: int
    phase_transitions: int
    density_variance: float


class TopologyAnalyzer:
    """
    Analyzes topological structure of semantic space during navigation.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.curvature_threshold = 0.7  # Above this = high curvature
        self.collapse_threshold = 0.85  # Above this = collapse risk
        
    def analyze_local(
        self,
        position: np.ndarray,
        target: np.ndarray,
        history: List[np.ndarray],
        velocity: Optional[np.ndarray] = None
    ) -> TopologicalState:
        """
        Perform local analysis and return a TopologicalState object.
        """
        curvature = self.estimate_local_curvature(history)
        
        # Energy ~ distance from target (lower = better)
        # Assuming start was somewhat further, we normalize roughly
        target_dist = np.linalg.norm(target - position)
        energy = min(1.0, target_dist) # Simplified energy model
        
        density = self.estimate_density(position, history)
        
        # Collapse risk check
        risk, _ = self.detect_collapse_risk(position, target, history, velocity)
        near_collapse = risk > self.collapse_threshold
        
        return TopologicalState(
            position=position.copy(),
            energy=energy,
            curvature=curvature,
            density=density,
            near_collapse=near_collapse,
            entropy=curvature * energy
        )

    def estimate_local_curvature(
        self, 
        history: List[np.ndarray],
        window: int = 3
    ) -> float:
        """
        Estimate local curvature from recent trajectory.
        Uses direction changes as proxy for geodesic curvature.
        """
        if len(history) < 3:
            return 0.0
        
        # Use last `window` points
        recent = history[-window:] if len(history) >= window else history
        
        # Compute consecutive direction changes
        curvatures = []
        for i in range(len(recent) - 2):
            d1 = recent[i+1] - recent[i]
            d2 = recent[i+2] - recent[i+1]
            
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            
            if n1 > 1e-8 and n2 > 1e-8:
                cos_angle = np.dot(d1, d2) / (n1 * n2)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                # Convert to curvature: 1 - cos = 0 for straight, 2 for 180°
                curvature = (1.0 - cos_angle) / 2.0
                curvatures.append(curvature)
        
        if curvatures:
            return float(np.max(curvatures))  # Return max curvature
        return 0.0
    
    def estimate_curvature_between(self, start: np.ndarray, target: np.ndarray) -> float:
        """Estimate curvature between two points (mock/heuristic without manifold)."""
        # In a real manifold, we'd use the metric tensor.
        # Here we use distance vs cosine dissimilarity as a proxy for curvature
        dist = np.linalg.norm(start - target)
        if dist < 1e-9: return 0.0
        
        cos_sim = np.dot(start, target) / (np.linalg.norm(start) * np.linalg.norm(target) + 1e-9)
        # If space is flat, dist relates to cos_sim linearly. Deviation implies curvature.
        # This is a heuristic placeholder.
        return max(0.0, 1.0 - cos_sim) * 0.5 

    def estimate_density(
        self,
        position: np.ndarray,
        known_points: Optional[List[np.ndarray]] = None,
        radius: float = 0.3
    ) -> float:
        """
        Estimate conceptual density around a position.
        """
        if known_points is None or len(known_points) < 2:
            return 0.5  # Unknown, assume medium
        
        # Count points within radius
        distances = [np.linalg.norm(position - p) for p in known_points]
        nearby = sum(1 for d in distances if d < radius)
        
        # Normalize by expected number
        expected = len(known_points) * 0.2  # Expect ~20% within radius
        density = min(1.0, nearby / (expected + 1e-8))
        
        return density
    
    def detect_collapse_risk(
        self,
        position: np.ndarray,
        target: np.ndarray,
        history: List[np.ndarray],
        velocity: Optional[np.ndarray] = None
    ) -> Tuple[float, str]:
        """
        Detect risk of topological collapse.
        """
        risk_factors = []
        reasons = []
        
        # Factor 1: High curvature
        curvature = self.estimate_local_curvature(history)
        if curvature > self.curvature_threshold:
            risk_factors.append(curvature)
            reasons.append("high_curvature")
        
        # Factor 2: Moving away from target
        if len(history) >= 2:
            prev_dist = np.linalg.norm(target - history[-1])
            curr_dist = np.linalg.norm(target - position)
            if curr_dist > prev_dist * 1.1:  # Moving away
                divergence = (curr_dist - prev_dist) / (prev_dist + 1e-8)
                risk_factors.append(min(1.0, divergence))
                reasons.append("diverging")
        
        # Factor 3: Oscillation detection
        oscillation = self._detect_oscillation(history)
        if oscillation > 0.5:
            risk_factors.append(oscillation)
            reasons.append("oscillation")
        
        if risk_factors:
            risk = float(np.max(risk_factors))
            reason = "+".join(reasons)
        else:
            risk = 0.0
            reason = "stable"
        
        return risk, reason
    
    def _detect_oscillation(self, history: List[np.ndarray]) -> float:
        """Detect oscillation pattern in trajectory."""
        if len(history) < 4:
            return 0.0
        
        # Check if alternating directions
        # Simplified for brevity
        return 0.0

