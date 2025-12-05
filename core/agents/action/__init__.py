"""
Alexandria - Action Agent Module
Refactored modular structure for Action Agent system.

This module provides the public interface for the Action Agent,
which executes actions, tests hypotheses, and registers evidence.

Usage:
    from core.agents.action import (
        ActionAgent,
        ActionType,
        ActionStatus,
        EvidenceType,
        create_action_agent_system
    )
    
    # Create system
    agent, simulator, registrar = create_action_agent_system(sfs_instance)
    
    # Execute action
    result = agent.execute_action(
        ActionType.PARAMETER_ADJUSTMENT,
        {"parameter_name": "V11_BETA", "new_value": 2.5}
    )
"""

# Main classes
from .agent import ActionAgent, create_action_agent_system
from .test_simulator import TestSimulator
from .evidence_registrar import EvidenceRegistrar

# Controllers
from .security_controller import SecurityController
from .parameter_controller import ParameterController

# Types
from .types import (
    ActionType,
    ActionStatus,
    EvidenceType,
    ActionResult,
    TestHypothesis
)

# Version info
__version__ = "2.0.0"
__refactored__ = True

# Public API
__all__ = [
    # Main orchestrator
    "ActionAgent",
    "create_action_agent_system",
    
    # Specialized modules
    "TestSimulator",
    "EvidenceRegistrar",
    
    # Controllers
    "SecurityController",
    "ParameterController",
    
    # Types
    "ActionType",
    "ActionStatus",
    "EvidenceType",
    "ActionResult",
    "TestHypothesis",
]
