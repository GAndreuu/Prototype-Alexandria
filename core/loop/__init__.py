"""
Alexandria Self-Feeding Loop
=============================

Módulo de ciclo auto-alimentado que conecta:
abduction → action → feedback → learning → vqvae → mycelial → abduction

Autor: Alexandria Project
Data: 2025-12-07
"""

from .hypothesis_executor import HypothesisExecutor
from .feedback_collector import ActionFeedbackCollector
from .incremental_learner import IncrementalLearner
from .self_feeding_loop import SelfFeedingLoop
from .loop_metrics import LoopMetrics

__all__ = [
    'HypothesisExecutor',
    'ActionFeedbackCollector', 
    'IncrementalLearner',
    'SelfFeedingLoop',
    'LoopMetrics'
]
