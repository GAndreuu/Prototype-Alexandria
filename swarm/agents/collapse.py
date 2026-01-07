"""
swarm/agents/collapse.py

CollapseAgent - High coherence, direct path seeking.

This agent operates in a "collapsed" cognitive state where all attention
is focused on the most direct path to the target. It suppresses alternatives
and noise, ideal for final convergence.

Inspiration: Flow state, laser focus, momentum-based convergence.
"""

import numpy as np
from typing import Optional

from .base import BaseAgent
from ..core import (
    NavigationStep, 
    HeuristicPersonality,
    NeurodiverseProfile, 
    TopologicalState, 
    NeurotypeName,
    Context,
    COLLAPSE_PROFILE
)


class CollapseAgent(BaseAgent):
    """
    High-coherence agent that seeks the most direct path.
    """
    
    def __init__(self, profile: Optional[NeurodiverseProfile] = None):
        self.profile = profile or COLLAPSE_PROFILE
        self.velocity = None  # Accumulated velocity for momentum
        self.momentum_alpha = 0.7  # Momentum coefficient
        
    def propose(self, ctx: Context) -> NavigationStep:
        """
        Propose a high-coherence step toward target.
        """
        # Direct vector to target
        direct = ctx.target_emb - ctx.current
        direct_norm = np.linalg.norm(direct)
        
        if direct_norm > 1e-8:
            direct = direct / direct_norm
        
        # Initialize or update velocity with momentum
        if self.velocity is None:
            self.velocity = direct.copy()
        else:
            # Momentum: blend previous velocity with current direction
            self.velocity = (
                self.momentum_alpha * self.velocity + 
                (1 - self.momentum_alpha) * direct
            )
            # Renormalize
            vel_norm = np.linalg.norm(self.velocity)
            if vel_norm > 1e-8:
                self.velocity = self.velocity / vel_norm
        
        # Apply coherence from profile (less noise with higher coherence)
        noise_scale = (1.0 - self.profile.coherence) * 0.1
        noise = np.random.randn(len(ctx.current)) * noise_scale
        
        direction = self.velocity + noise
        
        # Confidence increases with:
        # 1. Progress toward target
        # 2. Coherence of profile
        # 3. Stability of direction
        progress = 1.0 - (direct_norm / (ctx.initial_dist + 1e-8))
        direction_stability = float(np.dot(self.velocity, direct))
        
        base_confidence = 0.7
        confidence = base_confidence + 0.2 * progress + 0.1 * direction_stability
        confidence = max(0.1, min(1.0, confidence * self.profile.coherence))
        
        # --- Topological Awareness (Integrated from V3) ---
        topo_state = self._estimate_topology(ctx)
        
        # Adjust confidence based on collapse risk
        reasoning = f"Collapse mode (momentum={self.momentum_alpha:.2f}, progress={progress:.2%})"
        
        if topo_state.near_collapse:
            # Back off slightly if near actual collapse
            confidence = confidence * 0.7
            reasoning += " [CAUTION: near collapse]"
            
        return NavigationStep(
            agent_id="collapse",
            neurotype=NeurotypeName.COLLAPSE,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            topological_state=topo_state,
            personality=HeuristicPersonality.DIRECT_PATH # Legacy support
        )
    
    def _estimate_topology(self, ctx: Context) -> TopologicalState:
        """Estimate local topological state."""
        # Simple curvature estimate from direction changes
        if len(ctx.history) >= 2:
            prev_dir = ctx.history[-1] - ctx.history[-2]
            curr_dir = ctx.current - ctx.history[-1]
            
            # Curvature ~ angle between consecutive directions
            cos_angle = np.dot(prev_dir, curr_dir) / (
                np.linalg.norm(prev_dir) * np.linalg.norm(curr_dir) + 1e-8
            )
            curvature = 1.0 - max(-1.0, min(1.0, cos_angle))
        else:
            curvature = 0.0
        
        # Energy ~ distance from target (lower = better)
        dist = np.linalg.norm(ctx.target_emb - ctx.current)
        energy = dist / (ctx.initial_dist + 1e-8)
        
        # Density estimate (placeholder - would use actual KNN)
        density = 0.5  # Assume medium density
        
        # Near collapse if high curvature + high energy
        near_collapse = curvature > 0.8 and energy > 0.5
        
        return TopologicalState(
            position=ctx.current.copy(),
            energy=energy,
            curvature=curvature,
            density=density,
            near_collapse=near_collapse,
            entropy=curvature * energy  # Simple entropy proxy
        )
    
    def reset(self):
        """Reset agent state between navigations."""
        self.velocity = None
