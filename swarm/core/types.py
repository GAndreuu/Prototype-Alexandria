"""
swarm/core.py

Core definitions, types and interfaces for the Alexandria Swarm System.
Consolidates all versioned types into a single definitive source.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum

# =============================================================================
# ENUMS
# =============================================================================

class NeurotypeName(Enum):
    """Neurotype identifiers for agents."""
    COLLAPSE = "collapse"       # High coherence, direct path
    CRITICAL = "critical"       # Edge of chaos, phase transitions
    PSYCH = "psychedelic"       # Hyperconnectivity, non-linear
    AUTISTIC = "autistic"       # Hyperfocus, pattern detection
    RELAXED = "relaxed"         # Low coherence, broad exploration
    BALANCED = "balanced"       # Default balanced mode

class NavigationMode(Enum):
    """Operational modes for the Swarm."""
    SPRINT = "sprint"           # Fast, heuristic, low cost
    BALANCED = "balanced"       # Standard trade-off
    CAUTIOUS = "cautious"       # High fidelity, careful steps
    CREATIVE = "creative"       # High temperature, wide exploration

class HeuristicPersonality(Enum):
    """Legacy personality types (kept for compatibility)."""
    DIRECT_PATH = "direct"
    GRADIENT = "gradient"
    INTERPOLATION = "interpolation"
    MOMENTUM = "momentum"
    EXPLORER = "explorer"
    MYCELIAL_BRIDGE = "mycelial_bridge"

class ActionType(Enum):
    """Types of actions the Swarm can execute."""
    BRIDGE_CONCEPTS = "bridge_concepts"
    EXPLORE_CLUSTER = "explore_cluster"
    DEEPEN_TOPIC = "deepen_topic"

# =============================================================================
# DATA STRUCTURES
# DATA STRUCTURES
# =============================================================================

@dataclass
class SwarmAction:
    """Structured action command for the Swarm."""
    type: ActionType
    start: Union[str, np.ndarray]
    target: Union[str, np.ndarray]
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(np.random.randint(100000)))

@dataclass
class ConsensusResult:
    """Result of neurodiverse consensus."""
    direction: np.ndarray
    confidence: float
    reasoning: str
    contributing_neurotypes: Dict[NeurotypeName, float]
    # topological_state: Optional[TopologicalState] = None

@dataclass
class NeurodiverseProfile:
    """Cognitive profile for an agent."""
    name: NeurotypeName
    coherence: float = 0.5
    exploration: float = 0.5
    temperature: float = 1.0
    lateral_inhibition: float = 0.5
    integration_delay: int = 0
    
    def __post_init__(self):
        self.coherence = max(0.0, min(1.0, self.coherence))
        self.exploration = max(0.0, min(1.0, self.exploration))
        self.temperature = max(0.01, min(10.0, self.temperature))

@dataclass
class TopologicalState:
    """Topological state of current location."""
    position: np.ndarray
    energy: float = 0.0
    curvature: float = 0.0
    density: float = 0.0
    near_collapse: bool = False
    entropy: float = 0.0
    
    @property
    def stability(self) -> float:
        return 1.0 / (1.0 + self.energy + self.curvature)

@dataclass
class NavigationStep:
    """A proposed step by a single agent."""
    agent_id: str
    neurotype: NeurotypeName
    direction: np.ndarray
    confidence: float
    reasoning: str
    topological_state: Optional[TopologicalState] = None
    # For legacy support, optional:
    personality: Optional[HeuristicPersonality] = None
    alternative_directions: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        norm = np.linalg.norm(self.direction)
        if norm > 1e-9:
            self.direction = self.direction / norm

@dataclass
class Context:
    """Runtime context for navigation."""
    start_emb: np.ndarray
    target_emb: np.ndarray
    current: np.ndarray
    history: List[np.ndarray]
    step: int
    initial_dist: float
    
    # Extended context
    topological_state: Optional[TopologicalState] = None
    active_neurotypes: Dict[NeurotypeName, float] = field(default_factory=dict)
    stagnation_counter: int = 0
    
    @property
    def progress(self) -> float:
        current_dist = np.linalg.norm(self.target_emb - self.current)
        return 1.0 - (current_dist / (self.initial_dist + 1e-9))
    
    @property
    def current_similarity(self) -> float:
        dot = np.dot(self.current, self.target_emb)
        norm = np.linalg.norm(self.current) * np.linalg.norm(self.target_emb)
        return float(dot / (norm + 1e-9))

@dataclass
class NavigationResult:
    """Final result of a navigation session."""
    success: bool
    steps: int
    path: List[np.ndarray]
    init_similarity: float
    final_similarity: float
    improvement: float
    
    # Metadata
    neurotype_contributions: Dict[NeurotypeName, float] = field(default_factory=dict)
    topological_events: List[Dict] = field(default_factory=list)
    mode: NavigationMode = NavigationMode.BALANCED
    bridges_discovered: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class ModeConfig:
    """Configuration for a navigation mode."""
    max_steps: int
    min_confidence: float
    neurotype_mix: Dict[NeurotypeName, float]
    early_stopping_patience: int

# =============================================================================
# PROFILES
# =============================================================================

COLLAPSE_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.COLLAPSE,
    coherence=0.9,
    exploration=0.1,
    temperature=0.2,
    lateral_inhibition=0.9
)

CRITICAL_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.CRITICAL,
    coherence=0.6,
    exploration=0.6,
    temperature=1.2, # Near phase transition
    lateral_inhibition=0.4
)

PSYCHEDELIC_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.PSYCH,
    coherence=0.3, # Low coherence
    exploration=0.9,
    temperature=2.0, # High temperature
    lateral_inhibition=0.1 # Low inhibition = high connectivity
)

AUTISTIC_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.AUTISTIC,
    coherence=0.95,
    exploration=0.2,
    temperature=0.1,
    lateral_inhibition=0.8,
    integration_delay=2 # Takes time to process
)

RELAXED_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.RELAXED,
    coherence=0.4,
    exploration=0.7,
    temperature=1.0,
    lateral_inhibition=0.3
)

BALANCED_PROFILE = NeurodiverseProfile(
    name=NeurotypeName.BALANCED,
    coherence=0.5,
    exploration=0.5,
    temperature=0.8,
    lateral_inhibition=0.5
)

NEUROTYPE_PROFILES = {
    NeurotypeName.COLLAPSE: COLLAPSE_PROFILE,
    NeurotypeName.CRITICAL: CRITICAL_PROFILE,
    NeurotypeName.PSYCH: PSYCHEDELIC_PROFILE,
    NeurotypeName.AUTISTIC: AUTISTIC_PROFILE,
    NeurotypeName.RELAXED: RELAXED_PROFILE,
    NeurotypeName.BALANCED: BALANCED_PROFILE
}

def get_profile(name: Union[NeurotypeName, str]) -> NeurodiverseProfile:
    """Get profile by name."""
    if isinstance(name, str):
        try:
            name = NeurotypeName(name)
        except ValueError:
            return BALANCED_PROFILE
    return NEUROTYPE_PROFILES.get(name, BALANCED_PROFILE)
