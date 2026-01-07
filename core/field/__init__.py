"""
Campo Pré-Estrutural para Alexandria
=====================================

Este módulo implementa a camada de campo que opera no nível de potencial,
não de estrutura. É a cola geométrica que unifica Learning e Reasoning.

Componentes:
- DynamicManifold: Variedade diferenciável com dimensão variável
- RiemannianMetric: Métrica que deforma localmente
- FreeEnergyField: Campo F(x,t) sobre a variedade
- GeodesicFlow: Propagação por caminhos de menor energia (LEGADO)
- EfficientGeodesicComputer: Geodésicas otimizadas com PCA (RECOMENDADO)
- CycleDynamics: Expansão → Configuração → Compressão
- PreStructuralField: Wrapper unificado com conexões VQ-VAE/Mycelial

Uso:
    from core.field import PreStructuralField
    
    field = PreStructuralField()
    field.connect_vqvae(vqvae_model)
    field.connect_mycelial(mycelial)
    
    state = field.trigger(embedding)
    states = field.propagate(state, steps=5)
    graph = field.crystallize()
"""

from .manifold import DynamicManifold, ManifoldConfig, ManifoldPoint
from .metric import RiemannianMetric, MetricConfig
from .free_energy_field import FreeEnergyField, FieldConfig, FieldState
from .geodesic_flow import GeodesicFlow, GeodesicConfig  # Legado, usar EfficientGeodesicComputer
from .cycle_dynamics import CycleDynamics, CycleConfig, CycleState
from .pre_structural_field import PreStructuralField, PreStructuralConfig
from .efficient_geodesic import (
    EfficientGeodesicComputer, 
    EfficientGeodesicConfig, 
    GeodesicPath,
    create_efficient_geodesic
)

# =============================================================================
# ALIASES RECOMENDADOS
# =============================================================================
# EfficientGeodesicComputer é a versão otimizada de GeodesicFlow
# Usa redução PCA (384D → 64D) para performance 10-50x melhor
GeodesicFlowOptimized = EfficientGeodesicComputer

__all__ = [
    # Core components
    'DynamicManifold',
    'ManifoldConfig',
    'ManifoldPoint',
    'RiemannianMetric',
    'MetricConfig',
    'FreeEnergyField',
    'FieldConfig',
    'FieldState',
    'GeodesicFlow',  # Legado
    'GeodesicConfig',
    'CycleDynamics',
    'CycleConfig',
    'CycleState',
    # Unified wrapper
    'PreStructuralField',
    'PreStructuralConfig',
    # Efficient geodesic (RECOMENDADO)
    'EfficientGeodesicComputer',
    'EfficientGeodesicConfig',
    'GeodesicPath',
    'create_efficient_geodesic',
    # Aliases
    'GeodesicFlowOptimized',
]


