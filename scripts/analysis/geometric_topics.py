#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric & Information Geometry Topics para Alexandria
=======================================================

Tópicos para ingestão de papers relacionados ao Campo Pré-Estrutural.
Fundamentos matemáticos para implementação de:
- DynamicManifold
- RiemannianMetric
- FreeEnergyField
- GeodesicFlow

Autor: Alexandria Project
Data: 2025-12-08
"""

# ==========================
# GEOMETRIA DIFERENCIAL & INFORMATION GEOMETRY
# ==========================

GEOMETRIC_TOPICS = {
    # =========================================================================
    # 1. GEOMETRIA DIFERENCIAL BÁSICA
    # =========================================================================
    "differential_geometry_foundations": (
        '('
        '(all:"differential geometry" OR all:"smooth manifold" OR all:"differentiable manifold") '
        'AND '
        '(all:"tangent space" OR all:"cotangent" OR all:"vector bundle")'
        ')'
    ),
    
    "riemannian_geometry": (
        '('
        '(all:"Riemannian geometry" OR all:"Riemannian manifold" OR all:"metric tensor") '
        'AND '
        '(all:"curvature" OR all:"geodesic" OR all:"connection")'
        ')'
    ),
    
    "geodesic_equations": (
        '('
        '(all:"geodesic equation" OR all:"geodesic flow" OR all:"shortest path") '
        'AND '
        '(all:"manifold" OR all:"Riemannian" OR all:"metric")'
        ')'
    ),
    
    "curvature_tensors": (
        '('
        '(all:"Riemann curvature" OR all:"Ricci curvature" OR all:"scalar curvature") '
        'AND '
        '(all:"tensor" OR all:"manifold" OR all:"geometry")'
        ')'
    ),
    
    # =========================================================================
    # 2. INFORMATION GEOMETRY (Amari)
    # =========================================================================
    "information_geometry_core": (
        '('
        '(all:"information geometry" OR all:"Amari" OR all:"statistical manifold") '
        'AND '
        '(all:"Fisher information" OR all:"alpha-connection" OR all:"dual geometry")'
        ')'
    ),
    
    "fisher_information_metric": (
        '('
        '(all:"Fisher information metric" OR all:"Fisher-Rao" OR all:"Fisher matrix") '
        'AND '
        '(all:"statistical" OR all:"probability" OR all:"parametric")'
        ')'
    ),
    
    "natural_gradient_descent": (
        '('
        '(all:"natural gradient" OR all:"natural gradient descent" OR all:"Fisher preconditioner") '
        'AND '
        '(all:"optimization" OR all:"neural network" OR all:"deep learning")'
        ')'
    ),
    
    "exponential_family_geometry": (
        '('
        '(all:"exponential family" OR all:"exponential manifold" OR all:"e-geodesic") '
        'AND '
        '(all:"information geometry" OR all:"dual" OR all:"Bregman")'
        ')'
    ),
    
    # =========================================================================
    # 3. GEOMETRIA APLICADA A MACHINE LEARNING
    # =========================================================================
    "geometric_deep_learning": (
        '('
        '(all:"geometric deep learning" OR all:"manifold learning" OR all:"graph neural network") '
        'AND '
        '(all:"representation" OR all:"embedding" OR all:"equivariant")'
        ')'
    ),
    
    "neural_manifolds": (
        '('
        '(all:"neural manifold" OR all:"latent manifold" OR all:"representation geometry") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"embedding")'
        ')'
    ),
    
    "riemannian_optimization": (
        '('
        '(all:"Riemannian optimization" OR all:"optimization on manifolds" OR all:"manifold optimization") '
        'AND '
        '(all:"gradient descent" OR all:"constraint" OR all:"geometry")'
        ')'
    ),
    
    "wasserstein_geometry": (
        '('
        '(all:"Wasserstein distance" OR all:"optimal transport" OR all:"Wasserstein geometry") '
        'AND '
        '(all:"probability" OR all:"distribution" OR all:"machine learning")'
        ')'
    ),
    
    # =========================================================================
    # 4. FREE ENERGY & GEOMETRIA
    # =========================================================================
    "free_energy_geometry": (
        '('
        '(all:"free energy" OR all:"variational free energy" OR all:"Friston") '
        'AND '
        '(all:"geometry" OR all:"manifold" OR all:"information")'
        ')'
    ),
    
    "thermodynamic_geometry": (
        '('
        '(all:"thermodynamic geometry" OR all:"Ruppeiner geometry" OR all:"Weinhold geometry") '
        'AND '
        '(all:"metric" OR all:"curvature" OR all:"phase transition")'
        ')'
    ),
    
    "entropy_geometry": (
        '('
        '(all:"entropy" OR all:"maximum entropy" OR all:"entropy cone") '
        'AND '
        '(all:"geometry" OR all:"information" OR all:"manifold")'
        ')'
    ),
    
    # =========================================================================
    # 5. VARIEDADES DINÂMICAS E ADAPTATIVAS
    # =========================================================================
    "dynamic_manifolds": (
        '('
        '(all:"dynamic manifold" OR all:"evolving manifold" OR all:"manifold learning online") '
        'AND '
        '(all:"dimension" OR all:"adaptive" OR all:"streaming")'
        ')'
    ),
    
    "dimensionality_reduction_geometric": (
        '('
        '(all:"dimensionality reduction" OR all:"manifold embedding" OR all:"UMAP" OR all:"t-SNE") '
        'AND '
        '(all:"geometry" OR all:"preserving" OR all:"structure")'
        ')'
    ),
    
    "deformation_metrics": (
        '('
        '(all:"metric deformation" OR all:"metric learning" OR all:"adaptive metric") '
        'AND '
        '(all:"manifold" OR all:"Riemannian" OR all:"dynamic")'
        ')'
    ),
    
    # =========================================================================
    # 6. CONEXÕES COM FÍSICA
    # =========================================================================
    "general_relativity_geometry": (
        '('
        '(all:"general relativity" OR all:"spacetime" OR all:"Einstein equations") '
        'AND '
        '(all:"geometry" OR all:"metric" OR all:"curvature")'
        ')'
    ),
    
    "gauge_theory_neural": (
        '('
        '(all:"gauge theory" OR all:"fiber bundle" OR all:"connection") '
        'AND '
        '(all:"neural network" OR all:"machine learning" OR all:"equivariant")'
        ')'
    ),
    
    "symplectic_geometry": (
        '('
        '(all:"symplectic geometry" OR all:"Hamiltonian" OR all:"phase space") '
        'AND '
        '(all:"learning" OR all:"neural" OR all:"optimization")'
        ')'
    ),
    
    # =========================================================================
    # 7. TOPOLOGIA E ESTRUTURA
    # =========================================================================
    "topological_data_analysis": (
        '('
        '(all:"topological data analysis" OR all:"persistent homology" OR all:"TDA") '
        'AND '
        '(all:"machine learning" OR all:"manifold" OR all:"shape")'
        ')'
    ),
    
    "algebraic_topology_ml": (
        '('
        '(all:"algebraic topology" OR all:"homology" OR all:"cohomology") '
        'AND '
        '(all:"deep learning" OR all:"neural network" OR all:"representation")'
        ')'
    ),
    
    # =========================================================================
    # 8. IMPLEMENTAÇÃO COMPUTACIONAL
    # =========================================================================
    "computational_differential_geometry": (
        '('
        '(all:"computational geometry" OR all:"discrete differential geometry" OR all:"mesh") '
        'AND '
        '(all:"algorithm" OR all:"implementation" OR all:"numerical")'
        ')'
    ),
    
    "geodesic_computation": (
        '('
        '(all:"geodesic computation" OR all:"fast marching" OR all:"shortest path manifold") '
        'AND '
        '(all:"algorithm" OR all:"numerical" OR all:"efficient")'
        ')'
    ),
    
    "riemannian_neural_networks": (
        '('
        '(all:"Riemannian neural network" OR all:"hyperbolic neural network" OR all:"SPD neural network") '
        'AND '
        '(all:"geometry" OR all:"manifold" OR all:"representation")'
        ')'
    ),
}

# ==========================
# CATEGORIAS POR ÁREA
# ==========================

TOPIC_CATEGORIES = {
    "DIFFERENTIAL_GEOMETRY": [
        "differential_geometry_foundations",
        "riemannian_geometry",
        "geodesic_equations",
        "curvature_tensors",
    ],
    "INFORMATION_GEOMETRY": [
        "information_geometry_core",
        "fisher_information_metric",
        "natural_gradient_descent",
        "exponential_family_geometry",
    ],
    "GEOMETRIC_ML": [
        "geometric_deep_learning",
        "neural_manifolds",
        "riemannian_optimization",
        "wasserstein_geometry",
    ],
    "FREE_ENERGY_GEOMETRY": [
        "free_energy_geometry",
        "thermodynamic_geometry",
        "entropy_geometry",
    ],
    "DYNAMIC_MANIFOLDS": [
        "dynamic_manifolds",
        "dimensionality_reduction_geometric",
        "deformation_metrics",
    ],
    "PHYSICS_CONNECTIONS": [
        "general_relativity_geometry",
        "gauge_theory_neural",
        "symplectic_geometry",
    ],
    "TOPOLOGY": [
        "topological_data_analysis",
        "algebraic_topology_ml",
    ],
    "COMPUTATIONAL": [
        "computational_differential_geometry",
        "geodesic_computation",
        "riemannian_neural_networks",
    ],
}


# ==========================
# CONEXÃO COM CAMPO PRÉ-ESTRUTURAL
# ==========================

"""
COMO ESSES TÓPICOS IMPLEMENTAM O CAMPO PRÉ-ESTRUTURAL:

┌─────────────────────────────────────────────────────────────────┐
│                  PRE-STRUCTURAL FIELD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ DynamicManifold  │ ← differential_geometry_foundations       │
│  │                  │ ← dynamic_manifolds                       │
│  │                  │ ← dimensionality_reduction_geometric      │
│  └────────┬─────────┘                                           │
│           │                                                     │
│  ┌────────▼─────────┐                                           │
│  │ RiemannianMetric │ ← riemannian_geometry                     │
│  │                  │ ← fisher_information_metric               │
│  │                  │ ← deformation_metrics                     │
│  └────────┬─────────┘                                           │
│           │                                                     │
│  ┌────────▼─────────┐                                           │
│  │ FreeEnergyField  │ ← free_energy_geometry                    │
│  │                  │ ← thermodynamic_geometry                  │
│  │                  │ ← entropy_geometry                        │
│  └────────┬─────────┘                                           │
│           │                                                     │
│  ┌────────▼─────────┐                                           │
│  │  GeodesicFlow    │ ← geodesic_equations                      │
│  │                  │ ← geodesic_computation                    │
│  │                  │ ← natural_gradient_descent                │
│  └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PAPERS PRIORITÁRIOS:
1. Amari - "Information Geometry and Its Applications" (fundacional)
2. Friston - "A Free Energy Principle for the Brain" (unificação)
3. Bronstein - "Geometric Deep Learning" (aplicação prática)
4. Lee - "Introduction to Riemannian Manifolds" (matemática base)
"""


if __name__ == "__main__":
    print(f"Total de tópicos: {len(GEOMETRIC_TOPICS)}")
    print(f"Categorias: {len(TOPIC_CATEGORIES)}")
    print("\nTópicos por categoria:")
    for cat, topics in TOPIC_CATEGORIES.items():
        print(f"  {cat}: {len(topics)} tópicos")
