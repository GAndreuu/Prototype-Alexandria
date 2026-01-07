"""
swarm/agents/interpolation.py

InterpolationAgent - Smooth interpolation between start and target.
"""

import numpy as np
from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class InterpolationAgent(BaseAgent):
    """
    Interpolation agent - moves along a geodesic path from start to target.
    
    Uses linear interpolation: point_t = (1-t)*start + t*target
    """
    
    def propose(self, ctx: Context) -> NavigationStep:
        progress = ctx.step / 15.0
        target_point = (1 - progress) * ctx.start_emb + progress * ctx.target_emb
        direction = target_point - ctx.current
        
        return NavigationStep(
            agent_id="interpolation",
            personality=HeuristicPersonality.INTERPOLATION,
            direction=direction,
            confidence=0.6,
            reasoning=f"Interpolation ({progress:.0%})"
        )
