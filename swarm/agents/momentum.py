"""
swarm/agents/momentum.py

MomentumAgent - Uses accumulated momentum to overcome local minima.
"""

import numpy as np
from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class MomentumAgent(BaseAgent):
    """
    Momentum agent - combines previous movement direction with target direction.
    
    Formula: d_t = α * v_{t-1} + (1-α) * target_dir
    
    Helps overcome shallow local minima and reduces oscillations.
    """
    
    def propose(self, ctx: Context) -> NavigationStep:
        if len(ctx.history) < 2:
            return NavigationStep(
                agent_id="momentum",
                personality=HeuristicPersonality.MOMENTUM,
                direction=ctx.target_emb - ctx.current,
                confidence=0.4,
                reasoning="No momentum yet",
                neurotype=None
            )
        
        momentum = ctx.current - ctx.history[-2]
        target_dir = ctx.target_emb - ctx.current
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        
        combined = 0.3 * momentum + 0.7 * target_dir
        
        return NavigationStep(
            agent_id="momentum",
            personality=HeuristicPersonality.MOMENTUM,
            direction=combined,
            confidence=0.55,
            reasoning="Momentum + target",
            neurotype=None
        )
