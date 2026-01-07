"""
swarm/agents/explorer.py

ExplorerAgent - Stochastic exploration with directed noise.
"""

import numpy as np
from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class ExplorerAgent(BaseAgent):
    """
    Explorer agent - introduces directed randomness for exploration.
    
    Prevents stagnation at saddle points by adding noise to the direction.
    """
    
    def propose(self, ctx: Context) -> NavigationStep:
        random_dir = np.random.randn(384)
        target_dir = ctx.target_emb - ctx.current
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        
        direction = 0.2 * random_dir + 0.8 * target_dir
        
        return NavigationStep(
            agent_id="explorer",
            personality=HeuristicPersonality.EXPLORER,
            direction=direction,
            confidence=0.3,
            reasoning="Exploration",
            neurotype=None
        )
