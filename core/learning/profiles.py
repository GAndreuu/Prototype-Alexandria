"""
Alexandria Reasoning Profiles
=============================

Defines distinct cognitive personalities ("Laws") for the Multi-Agent Nemesis architecture.
Each profile represents a different strategy for traversing and updating the shared Mycelial memory.

Profiles:
1. THE SCOUT (Explorer): Fast, high novelty bias, low risk aversion. Expands the frontier.
2. THE JUDGE (Verifier): Slow, zero novelty, high risk aversion, deep planning. Validates connections.
3. THE WEAVER (Connector): Focuses on linking disconnected clusters (path gaps).

Autor: G (Alexandria Project)
Versão: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum, auto

@dataclass
class ReasoningProfile:
    """Configuration for a specific cognitive personality"""
    name: str
    description: str
    
    # Active Inference Parameters
    risk_weight: float          # Preference for known healthy states
    ambiguity_weight: float     # Preference for reducing uncertainty
    novelty_bonus: float        # Incentive for exploration
    
    # Planning
    planning_horizon: int       # Steps to look ahead
    temperature: float          # Softmax temp (higher = more random)
    
    # Plasticity (Meta-Hebbian)
    learning_rate_mod: float    # Multiplier for plasticity
    
    # Resource Usage
    max_steps_per_cycle: int    # Operations per "turn"

# =============================================================================
# PRE-DEFINED PROFILES
# =============================================================================

def get_scout_profile() -> ReasoningProfile:
    """
    The Scout: Cheap, fast, exploratory.
    Role: Rapidly generate hypotheses and find new papers/concepts.
    """
    return ReasoningProfile(
        name="The Scout",
        description="High-speed explorer of the unknown.",
        risk_weight=0.1,         # Low fear of being wrong
        ambiguity_weight=0.5,    # Moderate interest in clarity
        novelty_bonus=2.0,       # High drive for new things
        planning_horizon=2,      # Short-sighted (tactical)
        temperature=2.0,         # High randomness
        learning_rate_mod=1.5,   # Fast learner (maybe forgetful)
        max_steps_per_cycle=20
    )

def get_judge_profile() -> ReasoningProfile:
    """
    The Judge: Expensive, slow, critical.
    Role: Verify existing connections and remove weak ones.
    """
    return ReasoningProfile(
        name="The Judge",
        description="Critical verifier of truth/consistency.",
        risk_weight=5.0,         # Hates being wrong
        ambiguity_weight=2.0,    # Must resolve uncertainty completely
        novelty_bonus=-0.5,      # Penalizes "shiny new things"
        planning_horizon=8,      # Deep strategic thought
        temperature=0.1,         # Deterministic
        learning_rate_mod=0.2,   # Hard to change mind
        max_steps_per_cycle=5
    )

def get_weaver_profile() -> ReasoningProfile:
    """
    The Weaver: Structural, holistic.
    Role: Find structural gaps and bridge clusters.
    """
    return ReasoningProfile(
        name="The Weaver",
        description="Architect of long-range connections.",
        risk_weight=1.0,
        ambiguity_weight=1.0,
        novelty_bonus=0.5,       # Balanced
        planning_horizon=5,
        temperature=0.8,
        learning_rate_mod=1.0,
        max_steps_per_cycle=10
    )

ALL_PROFILES = {
    'scout': get_scout_profile,
    'judge': get_judge_profile,
    'weaver': get_weaver_profile
}
