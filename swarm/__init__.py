"""
Swarm Navigation System
=======================

Autonomous, neurodiverse navigation for high-dimensional semantic spaces.

Key Components:
- SwarmNavigator: Main engine suitable for all queries
- NeurodiverseConsensus: Fair voting system
- PersistentTopologicalMemory: Learning from past traversals
"""

from .core import (
    Context, 
    NavigationStep, 
    NavigationResult, 
    NeurotypeName, 
    NavigationMode, 
    TopologicalState
)

from .navigator import SwarmNavigator
from .core.consensus import NeurodiverseConsensus
from .core.memory import PersistentTopologicalMemory

try:
    from .mycelial import MycelialReasoningLite
except ImportError:
    # Optional dependency
    pass

__version__ = "3.1.0"
