"""
Campo Pré-Estrutural para Alexandria
=====================================

Este módulo implementa a camada de campo que opera no nível de potencial,
não de estrutura. É a cola geométrica que unifica Learning e Reasoning.

Componentes:
- DynamicManifold: Variedade diferenciável com dimensão variável
- RiemannianMetric: Métrica que deforma localmente
- FreeEnergyField: Campo F(x,t) sobre a variedade
- GeodesicFlow: Propagação por caminhos de menor energia
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
from .geodesic_flow import GeodesicFlow, GeodesicConfig
from .cycle_dynamics import CycleDynamics, CycleConfig, CycleState
from .pre_structural_field import PreStructuralField, PreStructuralConfig

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
    'GeodesicFlow',
    'GeodesicConfig',
    'CycleDynamics',
    'CycleConfig',
    'CycleState',
    # Unified wrapper
    'PreStructuralField',
    'PreStructuralConfig',
]

