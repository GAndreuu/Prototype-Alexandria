"""
swarm/agents/direct.py

DirectAgent - Goes straight to target.
"""

import numpy as np
from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class DirectAgent(BaseAgent):
    """
    Direct path agent - always points toward the target.
    
    Simple but effective in convex spaces. Confidence increases with progress.
    """
    
    def propose(self, ctx: Context) -> NavigationStep:
        direction = ctx.target_emb - ctx.current
        dist = np.linalg.norm(direction)
        progress = 1.0 - (dist / (ctx.initial_dist + 1e-8))
        
        return NavigationStep(
            agent_id="direct",
            personality=HeuristicPersonality.DIRECT_PATH,
            direction=direction,
            confidence=0.8 + 0.2 * progress,
            reasoning=f"Direct path (dist={dist:.4f})",
            neurotype=None  # Will be mapped by navigator or set explicitly
        )
