"""
swarm/agents/psychedelic.py

PsychedelicAgent - Hyperconnectivity and non-linear exploration.

This agent operates in a diffuse, highly exploratory state inspired by
the effects of psychedelics on brain connectivity. It creates unlikely
connections and explores the space in non-linear ways.

Inspiration: Psilocybin research, default mode network dissolution,
             increased global connectivity, reduced modularity.
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
    PSYCHEDELIC_PROFILE
)


class PsychedelicAgent(BaseAgent):
    """
    Hyperconnectivity agent for non-linear exploration.
    """
    
    def __init__(self, profile: Optional[NeurodiverseProfile] = None):
        self.profile = profile or PSYCHEDELIC_PROFILE
        self.activation_history: List[np.ndarray] = []
        self.fusion_memory: List[np.ndarray] = []  # Remember "interesting" points
        
    def propose(self, ctx: Context) -> NavigationStep:
        """
        Propose a hyperconnected, non-linear step.
        """
        # Basic direction to target
        direct = ctx.target_emb - ctx.current
        direct_norm = np.linalg.norm(direct)
        
        if direct_norm > 1e-8:
            direct = direct / direct_norm
        
        # Direction from start (for triangulation)
        from_start = ctx.current - ctx.start_emb
        from_start_norm = np.linalg.norm(from_start)
        if from_start_norm > 1e-8:
            from_start = from_start / from_start_norm
        
        # Generate multiple "hallucinated" directions
        directions = [direct]
        
        # 1. Perpendicular exploration (lateral thinking)
        perp = self._perpendicular_direction(direct, seed=ctx.step)
        directions.append(perp)
        
        # 2. Anti-gravity from start (escape familiar territory)
        directions.append(from_start * 0.5 + direct * 0.5)
        
        # 3. Random high-temperature direction
        random_dir = np.random.randn(len(ctx.current))
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)
        directions.append(random_dir)
        
        # 4. Midpoint direction (toward conceptual bridge)
        midpoint = (ctx.start_emb + ctx.target_emb) / 2
        midpoint = midpoint / (np.linalg.norm(midpoint) + 1e-8)
        to_midpoint = midpoint - ctx.current
        to_midpoint_norm = np.linalg.norm(to_midpoint)
        if to_midpoint_norm > 1e-8:
            to_midpoint = to_midpoint / to_midpoint_norm
        directions.append(to_midpoint)
        
        # Combine with high-temperature softmax-like weighting
        temperature = self.profile.temperature
        
        # Weight by alignment with target (but softened by temperature)
        weights = []
        for d in directions:
            alignment = float(np.dot(d, direct))
            # Higher temperature = more uniform weights
            weight = np.exp(alignment / temperature)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        # Sample or blend (blend for smooth exploration)
        final_direction = np.zeros_like(direct)
        for w, d in zip(weights, directions):
            final_direction += w * d
        
        # Normalize
        final_norm = np.linalg.norm(final_direction)
        if final_norm > 1e-8:
            final_direction = final_direction / final_norm
        
        # Add final noise based on exploration parameter
        noise = np.random.randn(len(final_direction)) * self.profile.exploration * 0.2
        final_direction = final_direction + noise
        final_direction = final_direction / (np.linalg.norm(final_direction) + 1e-8)
        
        # Store activation for pattern detection
        self.activation_history.append(final_direction.copy())
        if len(self.activation_history) > 10:
            self.activation_history.pop(0)
        
        # Confidence is moderate - we're exploring, not converging
        confidence = 0.3 + 0.2 * (1.0 - self.profile.exploration)
        
        # --- Topological Awareness ---
        topo_state = self._compute_psychedelic_topology(ctx)
        
        # Alternatives
        alternatives = self._generate_diverse_alternatives(ctx, 5)
        
        # Boost confidence if bridge pattern detected
        reasoning = f"Psychedelic mode (temp={temperature:.1f}, fusing {len(directions)} dirs)"
        if self._detected_bridge_pattern(ctx):
            confidence = min(0.7, confidence + 0.2)
            reasoning += " [BRIDGE DETECTED]"
        
        return NavigationStep(
            agent_id="psychedelic",
            neurotype=NeurotypeName.PSYCH,
            direction=final_direction,
            confidence=confidence,
            reasoning=reasoning,
            personality=HeuristicPersonality.EXPLORER,
            topological_state=topo_state,
            alternative_directions=alternatives
        )
    
    def _perpendicular_direction(self, base: np.ndarray, seed: int) -> np.ndarray:
        """Generate a direction perpendicular to base."""
        np.random.seed(seed * 42)
        random = np.random.randn(len(base))
        # Gram-Schmidt orthogonalization
        perp = random - np.dot(random, base) * base
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-8:
            return perp / perp_norm
        return random / (np.linalg.norm(random) + 1e-8)
    
    def _compute_psychedelic_topology(self, ctx: Context) -> TopologicalState:
        """
        Compute topology from psychedelic perspective.
        In this mode, we perceive space as more "fluid" with lower barriers.
        """
        dist = np.linalg.norm(ctx.target_emb - ctx.current)
        
        # Psychedelic perception: reduced energy barriers
        energy = (dist / (ctx.initial_dist + 1e-8)) * 0.7  # Perception of lower energy
        
        # High entropy perception (everything seems connected)
        entropy = 0.7 + 0.2 * np.random.random()
        
        # Curvature feels lower (space feels flatter, more connected)
        curvature = 0.3
        
        # High perceived density (many associations)
        density = 0.8
        
        return TopologicalState(
            position=ctx.current.copy(),
            energy=energy,
            curvature=curvature,
            density=density,
            near_collapse=False,  # Psychedelic mode doesn't fear collapse
            entropy=entropy
        )
    
    def _generate_diverse_alternatives(
        self, 
        ctx: Context, 
        n: int
    ) -> List[np.ndarray]:
        """Generate n diverse alternative directions."""
        alternatives = []
        
        for i in range(n):
            angle = 2 * np.pi * i / n
            
            # Create rotation in high-dim space (simplified)
            base = ctx.target_emb - ctx.current
            base = base / (np.linalg.norm(base) + 1e-8)
            
            perp = self._perpendicular_direction(base, ctx.step * 100 + i)
            
            alt = np.cos(angle * 0.5) * base + np.sin(angle * 0.5) * perp
            alt = alt / (np.linalg.norm(alt) + 1e-8)
            
            alternatives.append(alt)
        
        return alternatives
    
    def _detected_bridge_pattern(self, ctx: Context) -> bool:
        """Detect if we're on a conceptual bridge."""
        if len(ctx.history) < 3:
            return False
        
        # Check if similarity is increasing
        current_sim = float(np.dot(ctx.current, ctx.target_emb))
        prev_sim = float(np.dot(ctx.history[-1], ctx.target_emb))
        
        # Check if we're not on direct path
        direct = ctx.target_emb - ctx.start_emb
        actual = ctx.current - ctx.start_emb
        
        direct_norm = np.linalg.norm(direct)
        actual_norm = np.linalg.norm(actual)
        
        if direct_norm > 1e-8 and actual_norm > 1e-8:
            path_deviation = 1.0 - float(np.dot(
                direct / direct_norm, 
                actual / actual_norm
            ))
        else:
            path_deviation = 0.0
        
        # Bridge pattern: improving similarity while deviating from direct path
        return current_sim > prev_sim and path_deviation > 0.2
    
    def reset(self):
        """Reset agent state."""
        self.activation_history = []
        self.fusion_memory = []
