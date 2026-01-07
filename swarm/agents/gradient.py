"""
swarm/agents/gradient.py

RealGradientAgent - Uses analytical cosine similarity gradient.
"""

import numpy as np
from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class RealGradientAgent(BaseAgent):
    """
    Analytical gradient agent for cosine similarity maximization.
    
    Derivation:
        cos(x,y) = (x·y) / (‖x‖·‖y‖)
        ∇ₓ cos(x,y) = y/(‖x‖·‖y‖) - (x·y)·x/(‖x‖³·‖y‖)
    
    This gives the direction of maximum similarity increase.
    """
    
    def propose(self, ctx: Context) -> NavigationStep:
        x = ctx.current
        y = ctx.target_emb
        
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        dot = np.dot(x, y)
        
        # Avoid division by zero
        if x_norm < 1e-8 or y_norm < 1e-8:
            gradient = y - x
        else:
            # ∇ₓ cos(x,y) = y/(‖x‖‖y‖) - (x·y)x/(‖x‖³‖y‖)
            term1 = y / (x_norm * y_norm)
            term2 = (dot * x) / (x_norm**3 * y_norm)
            gradient = term1 - term2
        
        # Calculate expected gain
        grad_norm = np.linalg.norm(gradient)
        
        return NavigationStep(
            agent_id="gradient",
            personality=HeuristicPersonality.GRADIENT,
            direction=gradient,
            confidence=0.75,
            reasoning=f"Analytical gradient (|∇|={grad_norm:.4f})",
            neurotype=None
        )
