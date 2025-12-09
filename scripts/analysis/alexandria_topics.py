"""
Alexandria Research Topics v2.0
Tópicos organizados por área de pesquisa para coleta do arXiv.
"""

ALEXANDRIA_TOPICS = {
    # ===========================================================================
    # FUNDAMENTOS GEOMÉTRICOS
    # ===========================================================================
    "information_geometry": (
        '('
        '(all:"information geometry" OR all:"Fisher information matrix") '
        'AND '
        '(all:"neural network" OR all:"deep learning" OR all:"machine learning")'
        ')'
    ),
    "riemannian_deep_learning": (
        '('
        '(all:"Riemannian geometry" OR all:"Riemannian manifold") '
        'AND '
        '(all:"generative model" OR all:"deep learning" OR all:"neural network")'
        ')'
    ),
    "natural_gradient": (
        '('
        '(all:"natural gradient" OR all:"natural gradient descent") '
        'AND '
        '(all:"neural network" OR all:"optimization")'
        ')'
    ),
    "geodesics_probability": (
        '('
        '(all:"geodesic" AND all:"probability distribution") '
        'OR '
        '(all:"Wasserstein" AND all:"gradient flow")'
        ')'
    ),
    "curvature_laplacian": (
        '('
        '(all:"curvature" AND all:"Laplacian" AND all:"eigenvalue") '
        'OR '
        '(all:"spectral geometry" AND all:"neural")'
        ')'
    ),

    # ===========================================================================
    # FREE ENERGY PRINCIPLE
    # ===========================================================================
    "free_energy_principle": (
        '('
        '(all:"free energy principle" OR all:"variational free energy") '
        'AND '
        '(all:"brain" OR all:"cognition" OR all:"perception")'
        ')'
    ),
    "active_inference": (
        '('
        '(all:"active inference") '
        'AND '
        '(all:"Friston" OR all:"behavior" OR all:"action selection" OR all:"agent")'
        ')'
    ),
    "markov_blankets": (
        '('
        '(all:"Markov blanket" OR all:"Markov boundary") '
        'AND '
        '(all:"free energy" OR all:"self-organization" OR all:"autonomy")'
        ')'
    ),
    "variational_inference_brain": (
        '('
        '(all:"variational inference" OR all:"Laplace approximation") '
        'AND '
        '(all:"brain" OR all:"perception" OR all:"Bayesian brain")'
        ')'
    ),

    # ===========================================================================
    # PREDICTIVE CODING
    # ===========================================================================
    "predictive_coding": (
        '('
        '(all:"predictive coding" OR all:"predictive processing") '
        'AND '
        '(all:"visual cortex" OR all:"perception" OR all:"neural")'
        ')'
    ),
    "bayesian_brain": (
        '('
        '(all:"Bayesian brain" OR all:"Bayesian inference" AND all:"perception") '
        'OR '
        '(all:"phantom percept" OR all:"perceptual inference")'
        ')'
    ),
    "hierarchical_predictive": (
        '('
        '(all:"hierarchical predictive" OR all:"hierarchical inference") '
        'AND '
        '(all:"neural" OR all:"cortex" OR all:"brain")'
        ')'
    ),
    "precision_weighting": (
        '('
        '(all:"precision weighting" OR all:"precision-weighted") '
        'AND '
        '(all:"predictive" OR all:"inference" OR all:"attention")'
        ')'
    ),

    # ===========================================================================
    # META-LEARNING / PLASTICITY
    # ===========================================================================
    "meta_hebbian": (
        '('
        '(all:"meta-learning" AND all:"Hebbian") '
        'OR '
        '(all:"Hebbian plasticity" AND all:"learning to learn")'
        ')'
    ),
    "differentiable_plasticity": (
        '('
        '(all:"differentiable plasticity") '
        'OR '
        '(all:"plastic neural network" AND all:"meta-learning")'
        ')'
    ),
    "neuroevolution": (
        '('
        '(all:"NEAT" OR all:"neuroevolution") '
        'OR '
        '(all:"evolving neural network" AND all:"topology")'
        ')'
    ),
    "backpropamine": (
        '('
        '(all:"self-modifying neural network") '
        'OR '
        '(all:"neuromodulation" AND all:"deep learning")'
        ')'
    ),
    "learning_to_learn": (
        '('
        '(all:"learning to learn" OR all:"meta-learning") '
        'AND '
        '(all:"neural network" OR all:"few-shot")'
        ')'
    ),

    # ===========================================================================
    # QUANTIZAÇÃO VETORIAL
    # ===========================================================================
    "vq_vae": (
        '('
        '(all:"VQ-VAE" OR all:"vector quantized variational autoencoder") '
        'OR '
        '(all:"discrete representation learning")'
        ')'
    ),
    "product_quantization": (
        '('
        '(all:"product quantization" OR all:"PQ") '
        'AND '
        '(all:"nearest neighbor" OR all:"similarity search")'
        ')'
    ),
    "residual_quantization": (
        '('
        '(all:"residual vector quantization" OR all:"RVQ") '
        'OR '
        '(all:"multi-stage quantization")'
        ')'
    ),
    "finite_scalar_quantization": (
        '('
        '(all:"finite scalar quantization" OR all:"FSQ") '
        'OR '
        '(all:"scalar quantization" AND all:"neural")'
        ')'
    ),

    # ===========================================================================
    # TOPOLOGIA E DINÂMICA
    # ===========================================================================
    "topological_data_analysis": (
        '('
        '(all:"topological data analysis" OR all:"TDA") '
        'AND '
        '(all:"machine learning" OR all:"neural network")'
        ')'
    ),
    "persistent_homology": (
        '('
        '(all:"persistent homology") '
        'AND '
        '(all:"neural" OR all:"deep learning" OR all:"representation")'
        ')'
    ),
    "attractor_dynamics": (
        '('
        '(all:"attractor dynamics" OR all:"attractor network") '
        'AND '
        '(all:"recurrent" OR all:"memory" OR all:"neural")'
        ')'
    ),
    "self_organized_criticality": (
        '('
        '(all:"self-organized criticality" OR all:"critical dynamics") '
        'AND '
        '(all:"neural" OR all:"brain" OR all:"network")'
        ')'
    ),
    "edge_of_chaos": (
        '('
        '(all:"edge of chaos") '
        'AND '
        '(all:"computation" OR all:"neural" OR all:"reservoir")'
        ')'
    ),

    # ===========================================================================
    # CONSCIÊNCIA / COGNIÇÃO
    # ===========================================================================
    "integrated_information_theory": (
        '('
        '(all:"integrated information theory" OR all:"IIT" AND all:"consciousness") '
        'OR '
        '(all:"Tononi" AND all:"phi")'
        ')'
    ),
    "global_workspace_theory": (
        '('
        '(all:"global workspace theory" OR all:"global workspace") '
        'AND '
        '(all:"consciousness" OR all:"attention" OR all:"cognition")'
        ')'
    ),
    "consciousness_prior": (
        '('
        '(all:"consciousness prior" OR all:"conscious processing") '
        'AND '
        '(all:"neural" OR all:"Bengio" OR all:"attention")'
        ')'
    ),
    "attention_schema": (
        '('
        '(all:"attention schema" OR all:"attention schema theory") '
        'OR '
        '(all:"Graziano" AND all:"consciousness")'
        ')'
    ),

    # ===========================================================================
    # AUTO-MODIFICAÇÃO
    # ===========================================================================
    "godel_machines": (
        '('
        '(all:"Godel machine" OR all:"Goedel machine") '
        'OR '
        '(all:"self-improving" AND all:"Schmidhuber")'
        ')'
    ),
    "self_referential_nn": (
        '('
        '(all:"self-referential neural network") '
        'OR '
        '(all:"self-modifying" AND all:"network")'
        ')'
    ),
    "program_synthesis": (
        '('
        '(all:"program synthesis" AND all:"neural") '
        'OR '
        '(all:"neural program induction")'
        ')'
    ),
    "neural_architecture_search": (
        '('
        '(all:"neural architecture search" OR all:"NAS") '
        'AND '
        '(all:"AutoML" OR all:"architecture optimization")'
        ')'
    ),
}

# Total de tópicos
print(f"[Alexandria Topics] {len(ALEXANDRIA_TOPICS)} tópicos de pesquisa carregados.")
