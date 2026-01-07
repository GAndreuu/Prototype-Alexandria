"""
Alexandria Integrations Package
===============================

Módulos de integração que conectam todos os subsistemas do Alexandria
em uma arquitetura unificada.

Uso:
    from core.integrations import AlexandriaCore
    
    core = AlexandriaCore.from_vqvae(vqvae_model)
    result = core.cognitive_cycle(observation, goal)
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# BASE CLASSES (sempre disponíveis)
# =============================================================================
from .base_integration import (
    BaseCompositionalIntegration,
    BaseIntegrationConfig,
    IntegrationMetrics,
    create_integration
)


# Imports condicionais para graceful degradation
_available_modules = []

try:
    from .alexandria_unified import AlexandriaCore, AlexandriaConfig, CognitiveCycleResult
    _available_modules.append('alexandria_unified')
except ImportError as e:
    logger.warning(f"Could not import alexandria_unified: {e}")
    AlexandriaCore = None

try:
    from .nemesis_bridge_integration import (
        NemesisBridgeIntegration,
        NemesisBridgeConfig,
        GeometricEFE,
        GeometricAction
    )
    _available_modules.append('nemesis_bridge')
except ImportError as e:
    logger.warning(f"Could not import nemesis_bridge_integration: {e}")
    NemesisBridgeIntegration = None

try:
    from .learning_field_integration import (
        LearningFieldIntegration,
        LearningFieldConfig,
        GeometricPredictiveCoding,
        GeometricActiveInference
    )
    _available_modules.append('learning_field')
except ImportError as e:
    logger.warning(f"Could not import learning_field_integration: {e}")
    LearningFieldIntegration = None

try:
    from .abduction_compositional_integration import (
        AbductionCompositionalIntegration,
        AbductionCompositionalConfig,
        GeometricGap,
        GeodesicHypothesis
    )
    _available_modules.append('abduction_compositional')
except ImportError as e:
    logger.warning(f"Could not import abduction_compositional_integration: {e}")
    AbductionCompositionalIntegration = None

try:
    from .agents_compositional_integration import (
        AgentsCompositionalIntegration,
        AgentsCompositionalConfig,
        GeometricActionAgent,
        GeometricOracle
    )
    _available_modules.append('agents_compositional')
except ImportError as e:
    logger.warning(f"Could not import agents_compositional_integration: {e}")
    AgentsCompositionalIntegration = None

try:
    from .loop_compositional_integration import (
        LoopCompositionalIntegration,
        LoopCompositionalConfig,
        LoopPhase,
        CycleResult
    )
    _available_modules.append('loop_compositional')
except ImportError as e:
    logger.warning(f"Could not import loop_compositional_integration: {e}")
    LoopCompositionalIntegration = None

try:
    from .geodesic_bridge_integration import (
        GeodesicBridgeIntegration,
        GeodesicBridgeConfig,
        SemanticPath,
        ActivationMap,
        GeodesicField,
        create_geodesic_bridge
    )
    _available_modules.append('geodesic_bridge')
except ImportError as e:
    logger.warning(f"Could not import geodesic_bridge_integration: {e}")
    GeodesicBridgeIntegration = None

try:
    from .swarm_integration import (
        SwarmIntegration,
        SwarmIntegrationConfig,
        SwarmModeRecommendation,
        IntegratedNavigationResult,
        create_swarm_integration
    )
    _available_modules.append('swarm')
except ImportError as e:
    logger.warning(f"Could not import swarm_integration: {e}")
    SwarmIntegration = None


def available_modules() -> list:
    """Retorna lista de módulos disponíveis."""
    return _available_modules.copy()


def health_check() -> dict:
    """Verifica saúde de todos os módulos."""
    return {
        'alexandria_core': AlexandriaCore is not None,
        'nemesis_bridge': NemesisBridgeIntegration is not None,
        'learning_field': LearningFieldIntegration is not None,
        'abduction': AbductionCompositionalIntegration is not None,
        'agents': AgentsCompositionalIntegration is not None,
        'loop': LoopCompositionalIntegration is not None,
        'geodesic_bridge': GeodesicBridgeIntegration is not None,
        'swarm': SwarmIntegration is not None,
        'total_available': len(_available_modules)
    }


__all__ = [
    # Base Classes (NEW)
    'BaseCompositionalIntegration',
    'BaseIntegrationConfig',
    'IntegrationMetrics',
    'create_integration',
    # Core
    'AlexandriaCore',
    'AlexandriaConfig',
    'CognitiveCycleResult',
    # Nemesis
    'NemesisBridgeIntegration',
    'NemesisBridgeConfig',
    'GeometricEFE',
    'GeometricAction',
    # Learning
    'LearningFieldIntegration',
    'GeometricPredictiveCoding',
    'GeometricActiveInference',
    # Abduction
    'AbductionCompositionalIntegration',
    'GeometricGap',
    'GeodesicHypothesis',
    # Agents
    'AgentsCompositionalIntegration',
    'GeometricActionAgent',
    'GeometricOracle',
    # Loop
    'LoopCompositionalIntegration',
    'LoopPhase',
    'CycleResult',
    # Geodesic Bridge
    'GeodesicBridgeIntegration',
    'GeodesicBridgeConfig',
    'SemanticPath',
    'ActivationMap',
    'GeodesicField',
    'create_geodesic_bridge',
    # Swarm Integration
    'SwarmIntegration',
    'SwarmIntegrationConfig',
    'SwarmModeRecommendation',
    'IntegratedNavigationResult',
    'create_swarm_integration',
    # Utils
    'available_modules',
    'health_check',
]

