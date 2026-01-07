"""
swarm/agents/critical.py

CriticalAgent - Edge of chaos navigation.

This agent operates at the critical point between order and chaos,
where complex systems exhibit maximum responsiveness and adaptability.
It balances exploitation with exploration.

Inspiration: Phase transitions, criticality, sandpile dynamics.
"""

import numpy as np
from typing import Optional, List

from .base import BaseAgent
from ..core import (
    NavigationStep, 
    HeuristicPersonality,
    NeurodiverseProfile, 
    TopologicalState, 
    NeurotypeName,
    Context,
    CRITICAL_PROFILE
)


class CriticalAgent(BaseAgent):
    """
    Edge-of-chaos agent that balances order and disorder.
    """
    
    def __init__(self, profile: Optional[NeurodiverseProfile] = None):
        self.profile = profile or CRITICAL_PROFILE
        self.direction_history: List[np.ndarray] = []
        self.criticality_threshold = 0.5
        
    def propose(self, ctx: Context) -> NavigationStep:
        """
        Propose a critically-balanced step.
        """
        # Direct vector to target
        direct = ctx.target_emb - ctx.current
        direct_norm = np.linalg.norm(direct)
        
        if direct_norm > 1e-8:
            direct = direct / direct_norm
        
        # Estimate local criticality
        criticality = self._estimate_criticality(ctx)
        
        # At high criticality (phase transition), increase exploration
        # At low criticality (stable region), focus more
        exploration_weight = 0.2 + 0.3 * criticality
        
        # Generate exploration direction (perpendicular + random)
        random_dir = np.random.randn(len(ctx.current))
        # Make partially orthogonal to direct path
        random_dir = random_dir - np.dot(random_dir, direct) * direct * 0.5
        random_norm = np.linalg.norm(random_dir)
        if random_norm > 1e-8:
            random_dir = random_dir / random_norm
        
        # Combine direct and exploration
        direction = (1 - exploration_weight) * direct + exploration_weight * random_dir
        
        # Store for criticality estimation
        self.direction_history.append(direction.copy())
        if len(self.direction_history) > 5:
            self.direction_history.pop(0)
        
        # Confidence depends on criticality (less confident at phase transitions)
        base_confidence = 0.6
        # Lower confidence when highly critical (uncertain territory)
        confidence = base_confidence * (1.0 - 0.3 * criticality)
        
        # --- Topological Awareness ---
        topo_state = self._compute_topological_state(ctx)
        
        # Generate alternative directions at critical points
        alternatives = []
        if topo_state.curvature > 0.5:
            # High curvature = multiple viable paths
            for i in range(3):
                alt = self._generate_alternative(ctx, direction, i)
                alternatives.append(alt)

        return NavigationStep(
            agent_id="critical",
            neurotype=NeurotypeName.CRITICAL,
            direction=direction,
            confidence=confidence,
            reasoning=f"Critical mode (criticality={criticality:.2f}, explore={exploration_weight:.2f})",
            personality=HeuristicPersonality.GRADIENT,  # Closest match
            topological_state=topo_state,
            alternative_directions=alternatives
        )
    
    def _estimate_criticality(self, ctx: Context) -> float:
        """
        Estimate how "critical" (edge of chaos) the current position is.
        """
        factors = []
        
        # Factor 1: Direction instability
        if len(self.direction_history) >= 2:
            # Variance in recent directions
            dir_matrix = np.array(self.direction_history)
            variance = np.var(dir_matrix, axis=0).mean()
            factors.append(min(1.0, variance * 10))  # Scale appropriately
        
        # Factor 2: Progress stagnation
        if len(ctx.history) >= 3:
            recent_progress = [
                np.linalg.norm(ctx.history[i+1] - ctx.history[i])
                for i in range(len(ctx.history)-1)
            ][-3:]
            if len(recent_progress) > 0:
                stagnation = 1.0 - min(1.0, np.mean(recent_progress) * 20)
                factors.append(stagnation)
        
        # Factor 3: Distance from midpoint (most critical in the middle)
        if ctx.initial_dist > 1e-8:
            current_dist = np.linalg.norm(ctx.target_emb - ctx.current)
            relative_position = current_dist / ctx.initial_dist
            # Peak criticality at midpoint (0.5)
            midpoint_factor = 1.0 - 2.0 * abs(relative_position - 0.5)
            factors.append(max(0.0, midpoint_factor))
        
        if factors:
            return float(np.mean(factors))
        return 0.5  # Default moderate criticality
    
    def _compute_topological_state(self, ctx: Context) -> TopologicalState:
        """Compute full topological state."""
        criticality = self._estimate_criticality(ctx)
        
        # Map criticality to topological properties
        curvature = criticality * 0.8  # High criticality = high curvature
        
        dist = np.linalg.norm(ctx.target_emb - ctx.current)
        energy = dist / (ctx.initial_dist + 1e-8)
        
        # Entropy is high at critical points
        entropy = criticality * (1.0 - abs(2 * energy - 1.0))
        
        return TopologicalState(
            position=ctx.current.copy(),
            energy=energy,
            curvature=curvature,
            density=0.5,  # Placeholder
            near_collapse=criticality > 0.8 and energy > 0.7,
            entropy=entropy
        )
    
    def _generate_alternative(
        self, 
        ctx: Context, 
        main_dir: np.ndarray, 
        seed: int
    ) -> np.ndarray:
        """Generate alternative direction for critical points."""
        np.random.seed(ctx.step * 1000 + seed)
        
        # Orthogonal component
        ortho = np.random.randn(len(main_dir))
        ortho = ortho - np.dot(ortho, main_dir) * main_dir
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm > 1e-8:
            ortho = ortho / ortho_norm
        
        # Blend with main direction
        angle = (seed + 1) * 0.3  # ~17, 34, 51 degrees
        alt = np.cos(angle) * main_dir + np.sin(angle) * ortho
        return alt / (np.linalg.norm(alt) + 1e-8)
    
    def reset(self):
        """Reset agent state."""
        self.direction_history = []
