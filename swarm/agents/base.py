"""
swarm/agents/base.py

Abstract base class for navigation agents.
"""

from abc import ABC, abstractmethod
from ..core import Context, NavigationStep


class BaseAgent(ABC):
    """
    Abstract base class for all navigation agents.
    
    Each agent proposes a direction and confidence based on its heuristic.
    """
    
    @abstractmethod
    def propose(self, ctx: Context) -> NavigationStep:
        """
        Propose a navigation step given the current context.
        
        Args:
            ctx: Current navigation context
            
        Returns:
            NavigationStep with direction, confidence, and reasoning
        """
        pass
