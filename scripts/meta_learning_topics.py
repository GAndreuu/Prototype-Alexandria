#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Learning Topics para Alexandria Mass Ingestion
====================================================

Tópicos fragmentados para reconstruir meta-learning a partir de fundamentos.
Cada tópico representa um COMPONENTE FUNDAMENTAL que, combinado, cria meta-learning.

Estrutura:
1. Meta-Learning Core (MAML, learning to learn)
2. Meta-Cognição (self-monitoring, introspection)
3. Bayesian Foundations (priors, posteriors, uncertainty)
4. Free Energy Principle (predictive processing, active inference)
5. Hebbian/Synaptic Plasticity (STDP, biological learning)
6. Information Geometry (Fisher metric, natural gradients)
7. Optimization Landscapes (bi-level optimization, loss surfaces)
8. Memory Systems (working memory, episodic memory)
9. Abstraction & Representation (disentanglement, compositionality)
10. Self-Organization (emergence, autopoiesis)

Autor: Alexandria Project
Data: 2025-12-08
"""

# ==========================
# META-LEARNING TOPICS
# ==========================

META_LEARNING_TOPICS = {
    # =========================================================================
    # 1. META-LEARNING CORE
    # =========================================================================
    "meta_learning_core": (
        '('
        '(all:"meta-learning" OR all:"learning to learn" OR all:"few-shot learning") '
        'AND '
        '(all:"MAML" OR all:"model-agnostic" OR all:"task adaptation" '
        ' OR all:"bi-level optimization" OR all:"inner loop" OR all:"outer loop")'
        ')'
    ),
    
    "meta_learning_architectures": (
        '('
        '(all:"prototypical networks" OR all:"matching networks" OR all:"siamese networks") '
        'OR '
        '(all:"memory-augmented neural network" OR all:"neural turing machine" OR all:"differentiable memory")'
        ')'
    ),
    
    # =========================================================================
    # 2. META-COGNITION
    # =========================================================================
    "metacognition_self_monitoring": (
        '('
        '(all:"metacognition" OR all:"metacognitive" OR all:"self-monitoring") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"machine learning" OR all:"cognitive")'
        ')'
    ),
    
    "introspection_self_explanation": (
        '('
        '(all:"introspection" OR all:"self-explanation" OR all:"explainable AI") '
        'AND '
        '(all:"language model" OR all:"LLM" OR all:"transformer" OR all:"attention")'
        ')'
    ),
    
    "confidence_calibration": (
        '('
        '(all:"confidence calibration" OR all:"uncertainty calibration" OR all:"epistemic uncertainty") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"prediction")'
        ')'
    ),
    
    # =========================================================================
    # 3. BAYESIAN FOUNDATIONS
    # =========================================================================
    "bayesian_meta_learning": (
        '('
        '(all:"Bayesian meta-learning" OR all:"PAC-Bayesian" OR all:"hyper-prior") '
        'OR '
        '(all:"Bayesian inference" AND all:"task distribution")'
        ')'
    ),
    
    "prior_learning": (
        '('
        '(all:"prior learning" OR all:"learned prior" OR all:"meta-prior") '
        'AND '
        '(all:"Bayesian" OR all:"probabilistic" OR all:"distribution")'
        ')'
    ),
    
    "variational_inference_foundations": (
        '('
        '(all:"variational inference" OR all:"variational Bayes" OR all:"ELBO") '
        'AND '
        '(all:"amortized inference" OR all:"approximate inference" OR all:"KL divergence")'
        ')'
    ),
    
    # =========================================================================
    # 4. FREE ENERGY PRINCIPLE (Friston)
    # =========================================================================
    "free_energy_principle": (
        '('
        '(all:"free energy principle" OR all:"variational free energy" OR all:"Friston") '
        'AND '
        '(all:"biological" OR all:"brain" OR all:"self-organization" OR all:"Markov blanket")'
        ')'
    ),
    
    "predictive_processing": (
        '('
        '(all:"predictive processing" OR all:"predictive coding" OR all:"prediction error") '
        'AND '
        '(all:"hierarchical" OR all:"generative model" OR all:"top-down" OR all:"bottom-up")'
        ')'
    ),
    
    "active_inference": (
        '('
        '(all:"active inference" OR all:"expected free energy" OR all:"epistemic value") '
        'AND '
        '(all:"action selection" OR all:"planning" OR all:"POMDP" OR all:"decision making")'
        ')'
    ),
    
    # =========================================================================
    # 5. HEBBIAN / SYNAPTIC PLASTICITY
    # =========================================================================
    "hebbian_learning": (
        '('
        '(all:"Hebbian learning" OR all:"Hebbian plasticity" OR all:"fire together wire together") '
        'AND '
        '(all:"neural network" OR all:"synapse" OR all:"learning rule")'
        ')'
    ),
    
    "stdp_spike_timing": (
        '('
        '(all:"spike-timing dependent plasticity" OR all:"STDP" OR all:"spike timing") '
        'AND '
        '(all:"synapse" OR all:"LTP" OR all:"LTD" OR all:"temporal")'
        ')'
    ),
    
    "meta_plasticity": (
        '('
        '(all:"metaplasticity" OR all:"meta-plasticity" OR all:"synaptic scaling") '
        'AND '
        '(all:"neural" OR all:"homeostatic" OR all:"BCM")'
        ')'
    ),
    
    # =========================================================================
    # 6. INFORMATION GEOMETRY
    # =========================================================================
    "information_geometry": (
        '('
        '(all:"information geometry" OR all:"Fisher information" OR all:"statistical manifold") '
        'AND '
        '(all:"Riemannian" OR all:"metric" OR all:"geometry")'
        ')'
    ),
    
    "natural_gradient": (
        '('
        '(all:"natural gradient" OR all:"natural gradient descent" OR all:"Fisher matrix") '
        'AND '
        '(all:"optimization" OR all:"neural network" OR all:"learning")'
        ')'
    ),
    
    "optimal_transport": (
        '('
        '(all:"optimal transport" OR all:"Wasserstein" OR all:"earth mover distance") '
        'AND '
        '(all:"machine learning" OR all:"distribution" OR all:"generative")'
        ')'
    ),
    
    # =========================================================================
    # 7. OPTIMIZATION LANDSCAPES
    # =========================================================================
    "loss_landscape": (
        '('
        '(all:"loss landscape" OR all:"loss surface" OR all:"optimization landscape") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"saddle point" OR all:"local minima")'
        ')'
    ),
    
    "bilevel_optimization": (
        '('
        '(all:"bilevel optimization" OR all:"bi-level" OR all:"nested optimization") '
        'AND '
        '(all:"machine learning" OR all:"hyperparameter" OR all:"meta")'
        ')'
    ),
    
    "implicit_differentiation": (
        '('
        '(all:"implicit differentiation" OR all:"implicit gradient" OR all:"implicit function theorem") '
        'AND '
        '(all:"optimization" OR all:"neural network" OR all:"learning")'
        ')'
    ),
    
    # =========================================================================
    # 8. MEMORY SYSTEMS
    # =========================================================================
    "working_memory_neural": (
        '('
        '(all:"working memory" AND (all:"neural network" OR all:"computational model" OR all:"prefrontal")) '
        'OR '
        '(all:"attractor network" AND all:"memory")'
        ')'
    ),
    
    "episodic_memory_ml": (
        '('
        '(all:"episodic memory" OR all:"experience replay" OR all:"memory buffer") '
        'AND '
        '(all:"reinforcement learning" OR all:"neural network" OR all:"retrieval")'
        ')'
    ),
    
    "memory_consolidation": (
        '('
        '(all:"memory consolidation" OR all:"catastrophic forgetting" OR all:"continual learning") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"elastic weight consolidation")'
        ')'
    ),
    
    # =========================================================================
    # 9. ABSTRACTION & REPRESENTATION
    # =========================================================================
    "disentanglement": (
        '('
        '(all:"disentangled representation" OR all:"disentanglement" OR all:"independent factors") '
        'AND '
        '(all:"VAE" OR all:"generative" OR all:"unsupervised")'
        ')'
    ),
    
    "compositionality": (
        '('
        '(all:"compositional generalization" OR all:"compositionality" OR all:"systematic generalization") '
        'AND '
        '(all:"neural network" OR all:"language" OR all:"reasoning")'
        ')'
    ),
    
    "abstraction_hierarchy": (
        '('
        '(all:"abstraction hierarchy" OR all:"hierarchical abstraction" OR all:"concept learning") '
        'AND '
        '(all:"neural" OR all:"cognitive" OR all:"representation")'
        ')'
    ),
    
    # =========================================================================
    # 10. SELF-ORGANIZATION & EMERGENCE
    # =========================================================================
    "autopoiesis": (
        '('
        '(all:"autopoiesis" OR all:"autopoietic" OR all:"self-producing") '
        'AND '
        '(all:"cognition" OR all:"living system" OR all:"self-organization")'
        ')'
    ),
    
    "emergence_complexity": (
        '('
        '(all:"emergence" OR all:"emergent behavior" OR all:"complex systems") '
        'AND '
        '(all:"neural" OR all:"self-organization" OR all:"criticality")'
        ')'
    ),
    
    "reservoir_computing": (
        '('
        '(all:"reservoir computing" OR all:"echo state network" OR all:"liquid state machine") '
        'AND '
        '(all:"temporal" OR all:"dynamics" OR all:"computation")'
        ')'
    ),
    
    # =========================================================================
    # 11. MATHEMATICAL FOUNDATIONS
    # =========================================================================
    "category_theory_ml": (
        '('
        '(all:"category theory" OR all:"categorical" OR all:"functor") '
        'AND '
        '(all:"machine learning" OR all:"neural network" OR all:"compositional")'
        ')'
    ),
    
    "dynamical_systems_learning": (
        '('
        '(all:"dynamical systems" OR all:"attractor" OR all:"bifurcation") '
        'AND '
        '(all:"neural network" OR all:"recurrent" OR all:"learning")'
        ')'
    ),
    
    "renormalization_group": (
        '('
        '(all:"renormalization group" OR all:"RG flow" OR all:"scaling") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"representation")'
        ')'
    ),
    
    # =========================================================================
    # 12. PHYSICS-INSPIRED LEARNING
    # =========================================================================
    "thermodynamic_learning": (
        '('
        '(all:"thermodynamic" OR all:"entropy production" OR all:"dissipation") '
        'AND '
        '(all:"learning" OR all:"neural" OR all:"self-organization")'
        ')'
    ),
    
    "energy_based_models": (
        '('
        '(all:"energy-based model" OR all:"Boltzmann machine" OR all:"Hopfield network") '
        'AND '
        '(all:"learning" OR all:"associative memory" OR all:"equilibrium")'
        ')'
    ),
    
    "quantum_inspired_ml": (
        '('
        '(all:"quantum machine learning" OR all:"quantum neural network" OR all:"quantum computing") '
        'AND '
        '(all:"optimization" OR all:"learning" OR all:"encoding")'
        ')'
    ),
    
    # =========================================================================
    # 13. FAILED PAPERS TOPICS (Extraídos do failed_papers_list.txt)
    # =========================================================================
    
    # Spiking Neural Networks / Neuromorphic
    "colanet_spiking": (
        '('
        '(all:"spiking neural network" OR all:"CoLaNET" OR all:"neuromorphic computing") '
        'AND '
        '(all:"digital" OR all:"algorithm" OR all:"simulation")'
        ')'
    ),
    
    # Geometric Field Theory / Transformers
    "geometric_transformers": (
        '('
        '(all:"geometric field theory" OR all:"manifold embeddings" OR all:"kernel modulation") '
        'AND '
        '(all:"transformer" OR all:"attention" OR all:"neural network")'
        ')'
    ),
    
    # AI Planning / Cyber-Physical Systems
    "ai_planning_cps": (
        '('
        '(all:"AI planning" OR all:"automated planning" OR all:"cyber-physical systems") '
        'AND '
        '(all:"CAIPI" OR all:"workshop" OR all:"proceedings")'
        ')'
    ),
    
    # Instruction Following / In-Context Learning
    "instruction_inference": (
        '('
        '(all:"instruction inference" OR all:"instruction following" OR all:"in-context learning") '
        'AND '
        '(all:"language model" OR all:"LLM" OR all:"reasoning")'
        ')'
    ),
    
    # Black Hole / Dark Matter / Gravitational Lensing
    "black_hole_dark_matter": (
        '('
        '(all:"black hole" OR all:"dark matter halo" OR all:"Dehnen profile") '
        'AND '
        '(all:"gravitational lensing" OR all:"thermodynamics" OR all:"light ring")'
        ')'
    ),
    
    # Sparse Dictionary Learning / Orthogonal
    "sparse_orthogonal_dictionary": (
        '('
        '(all:"sparse dictionary learning" OR all:"orthogonal dictionary" OR all:"sparse coding") '
        'AND '
        '(all:"exact" OR all:"learning" OR all:"representation")'
        ')'
    ),
    
    # Multi-View 3D Detection / BEV
    "multiview_3d_detection": (
        '('
        '(all:"multi-view 3D" OR all:"3D object detection" OR all:"PETR") '
        'AND '
        '(all:"quantized" OR all:"position embedding" OR all:"BEV")'
        ')'
    ),
    
    # Rule Learning / LLM Agents / Abduction
    "rule_learning_llm": (
        '('
        '(all:"rule learning" OR all:"induction deduction abduction" OR all:"IDEA") '
        'AND '
        '(all:"language model" OR all:"LLM agent" OR all:"reasoning")'
        ')'
    ),
    
    # Image Quality Assessment / Causal
    "iqa_causal": (
        '('
        '(all:"image quality assessment" OR all:"IQA" OR all:"perceptual") '
        'AND '
        '(all:"causal" OR all:"counterfactual" OR all:"abductive")'
        ')'
    ),
    
    # Control Barrier Functions / Neural
    "neural_control_barrier": (
        '('
        '(all:"control barrier function" OR all:"control barrier certificate" OR all:"monotone neural") '
        'AND '
        '(all:"neural network" OR all:"safety" OR all:"verification")'
        ')'
    ),
    
    # Implicit Planning / Procedural
    "implicit_planning": (
        '('
        '(all:"implicit planning" OR all:"procedural" OR all:"PARADISE") '
        'AND '
        '(all:"language model" OR all:"evaluation" OR all:"dataset")'
        ')'
    ),
    
    # Visual Puzzles / CoT Error Detection
    "visual_puzzles_cot": (
        '('
        '(all:"visual puzzle" OR all:"PRISM" OR all:"puzzle-based") '
        'AND '
        '(all:"chain-of-thought" OR all:"error detection" OR all:"reasoning")'
        ')'
    ),
    
    # Vector Symbolic / Probabilistic Abduction
    "vector_symbolic_abduction": (
        '('
        '(all:"vector symbolic architecture" OR all:"probabilistic abduction" OR all:"VSA") '
        'AND '
        '(all:"visual abstract reasoning" OR all:"learning rules" OR all:"representation")'
        ')'
    ),
    
    # Epistemic Logic / Rationality / Game Theory
    "epistemic_rationality": (
        '('
        '(all:"epistemic logic" OR all:"rationality" OR all:"TARK") '
        'AND '
        '(all:"game theory" OR all:"knowledge" OR all:"proceedings")'
        ')'
    ),
    
    # Knowledge Representation / Neuro-Symbolic
    "knowledge_representation_llm": (
        '('
        '(all:"knowledge representation" OR all:"neuro-symbolic" OR all:"NeLaMKRR") '
        'AND '
        '(all:"language model" OR all:"reasoning" OR all:"workshop")'
        ')'
    ),
    
    # Medical Dialogue / Diagnostic Reasoning
    "medical_diagnostic_reasoning": (
        '('
        '(all:"medical dialogue" OR all:"diagnostic reasoning" OR all:"clinical") '
        'AND '
        '(all:"reasoning" OR all:"doctor" OR all:"patient")'
        ')'
    ),
    
    # Compressive Sensing / Tikhonov / Spectral
    "compressive_sensing_reconstruction": (
        '('
        '(all:"compressive sensing" OR all:"Tikhonov" OR all:"spectral filtering") '
        'AND '
        '(all:"image reconstruction" OR all:"sparse" OR all:"dictionary")'
        ')'
    ),
    
    # Robust VQ-VAE
    "robust_vqvae": (
        '('
        '(all:"VQ-VAE" OR all:"vector quantized" OR all:"robust") '
        'AND '
        '(all:"variational autoencoder" OR all:"discrete" OR all:"representation")'
        ')'
    ),
    
    # Neuro-Symbolic Arithmetic Reasoning
    "neuro_symbolic_arithmetic": (
        '('
        '(all:"neuro-symbolic" OR all:"arithmetic reasoning" OR all:"abstract reasoning") '
        'AND '
        '(all:"LLM" OR all:"comparison" OR all:"learning to reason")'
        ')'
    ),
    
    # Finite-Strain Microelasticity
    "finite_strain_microelasticity": (
        '('
        '(all:"finite-strain" OR all:"microelasticity" OR all:"variationally consistent") '
        'AND '
        '(all:"framework" OR all:"microstructure" OR all:"mechanics")'
        ')'
    ),
}

# ==========================
# TOPIC CATEGORIES MAPPING
# ==========================

TOPIC_CATEGORIES = {
    "META_LEARNING_CORE": [
        "meta_learning_core",
        "meta_learning_architectures",
    ],
    "META_COGNITION": [
        "metacognition_self_monitoring",
        "introspection_self_explanation",
        "confidence_calibration",
    ],
    "BAYESIAN_FOUNDATIONS": [
        "bayesian_meta_learning",
        "prior_learning",
        "variational_inference_foundations",
    ],
    "FREE_ENERGY_PRINCIPLE": [
        "free_energy_principle",
        "predictive_processing",
        "active_inference",
    ],
    "HEBBIAN_PLASTICITY": [
        "hebbian_learning",
        "stdp_spike_timing",
        "meta_plasticity",
    ],
    "INFORMATION_GEOMETRY": [
        "information_geometry",
        "natural_gradient",
        "optimal_transport",
    ],
    "OPTIMIZATION": [
        "loss_landscape",
        "bilevel_optimization",
        "implicit_differentiation",
    ],
    "MEMORY_SYSTEMS": [
        "working_memory_neural",
        "episodic_memory_ml",
        "memory_consolidation",
    ],
    "ABSTRACTION": [
        "disentanglement",
        "compositionality",
        "abstraction_hierarchy",
    ],
    "SELF_ORGANIZATION": [
        "autopoiesis",
        "emergence_complexity",
        "reservoir_computing",
    ],
    "MATHEMATICAL_FOUNDATIONS": [
        "category_theory_ml",
        "dynamical_systems_learning",
        "renormalization_group",
    ],
    "PHYSICS_INSPIRED": [
        "thermodynamic_learning",
        "energy_based_models",
        "quantum_inspired_ml",
    ],
}


# ==========================
# SUMMARY
# ==========================

"""
COMO ESSES TÓPICOS RECONSTROEM META-LEARNING:

Meta-Learning = Capacidade de "aprender a aprender"

Fragmentação em componentes fundamentais:

┌─────────────────────────────────────────────────────────────────┐
│                     META-LEARNING                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   BAYESIAN   │  │    FREE      │  │  INFORMATION │          │
│  │  PRIORS      │→ │   ENERGY     │→ │   GEOMETRY   │          │
│  │  (incerteza) │  │  (previsão)  │  │  (gradientes)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                 ↓                  ↓                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   HEBBIAN    │  │   MEMORY     │  │ OPTIMIZATION │          │
│  │  PLASTICITY  │→ │   SYSTEMS    │→ │  LANDSCAPES  │          │
│  │  (sinapse)   │  │  (episódica) │  │  (bi-level)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                 ↓                  ↓                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ ABSTRACTION  │  │    SELF-     │  │    META-     │          │
│  │ REPRESENTATION│→│ ORGANIZATION │→ │  COGNITION   │          │
│  │ (disentangle)│  │ (emergence)  │  │ (monitoring) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

CONEXÃO COM ALEXANDRIA ATUAL:

1. Free Energy Principle → Já implementado em core/learning/free_energy.py
2. Predictive Coding → Já implementado em core/learning/predictive_coding.py
3. Active Inference → Já implementado em core/learning/active_inference.py
4. Hebbian Learning → Já implementado em MycelialReasoning
5. VQ-VAE (codebook) → Compressão discreta para memória eficiente
6. Meta-Hebbian → Já implementado em core/learning/meta_hebbian.py

PRÓXIMO PASSO (Meta-Cognição):
- Adicionar módulo de "self-monitoring" que observa os outros módulos
- Usar métricas do LoopMetrics + FreeEnergy trend para detectar "estados internos"
- Implementar "introspection" baseado em confidence calibration

"""

if __name__ == "__main__":
    print(f"Total de tópicos: {len(META_LEARNING_TOPICS)}")
    print(f"Categorias: {len(TOPIC_CATEGORIES)}")
    print("\nTópicos por categoria:")
    for cat, topics in TOPIC_CATEGORIES.items():
        print(f"  {cat}: {len(topics)} tópicos")
