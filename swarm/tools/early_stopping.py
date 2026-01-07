"""
swarm/tools/early_stopping.py

Early Stopping Committee
Consolidated from v3_1/early_stopping.py.

Multi-voter system that decides when to stop navigation early.
Instead of always using max_steps, uses multiple criteria:
1. ConvergenceVoter
2. ProgressVoter
3. ConfidenceVoter
4. EnergyVoter
5. TopologyVoter
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class Vote:
    """A single voter's decision"""
    voter_name: str
    should_stop: bool
    confidence: float  # 0-1, how confident in this vote
    reason: str

@dataclass
class StopDecision:
    """Final decision from committee"""
    should_stop: bool
    votes: List[Vote]
    agreement_ratio: float
    primary_reason: str
    success_prediction: bool = False # Added field


class BaseVoter(ABC):
    """Base class for all voters"""
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def vote(
        self, 
        trajectory: List[np.ndarray],
        target: np.ndarray,
        current_step: int,
        context: Dict[str, Any]
    ) -> Vote:
        """Cast a vote on whether to stop"""
        pass

class ConvergenceVoter(BaseVoter):
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
    
    @property
    def name(self) -> str: return "convergence"
    
    def vote(self, trajectory, target, current_step, context) -> Vote:
        if not trajectory:
            return Vote(self.name, False, 0.0, "No trajectory")
        current = trajectory[-1]
        similarity = self._cosine_similarity(current, target)
        if similarity >= self.threshold:
            return Vote(self.name, True, similarity, f"Converged: sim={similarity:.4f} >= {self.threshold}")
        return Vote(self.name, False, 1.0 - similarity, f"Not converged")

    def _cosine_similarity(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9: return 0.0
        return float(np.dot(a, b) / (na * nb))

# Simplified implementations for brevity in consolidation, reusing logic
class ProgressVoter(BaseVoter):
    def __init__(self, patience=5, min_progress=0.01):
        self.patience = patience
        self.min_progress = min_progress
    @property
    def name(self): return "progress"
    def vote(self, trajectory, target, current_step, context):
        if len(trajectory) < self.patience + 1: 
            return Vote(self.name, False, 0.0, "Insufficient history")
        
        # Calculate progress over last N steps
        recent_positions = trajectory[-self.patience:]
        distances_to_target = [np.linalg.norm(pos - target) for pos in recent_positions]
        
        # Progress is reduction in distance
        progress = distances_to_target[0] - distances_to_target[-1]
        
        if progress < self.min_progress:
            return Vote(self.name, True, 0.8, f"Stagnated: progress={progress:.4f}")
        return Vote(self.name, False, 0.2, f"Progressing: {progress:.4f}")


class PathQualityVoter(BaseVoter):
    """
    Votes based on path quality metrics: smoothness, progress, diversity.
    A high-quality path may warrant early stopping even without full convergence.
    """
    def __init__(self, min_quality: float = 0.75):
        self.min_quality = min_quality
    
    @property
    def name(self) -> str: return "path_quality"
    
    def vote(self, trajectory, target, current_step, context) -> Vote:
        if len(trajectory) < 4:
            return Vote(self.name, False, 0.0, "Path too short")
        
        quality = self._compute_quality(trajectory, target)
        
        if quality >= self.min_quality:
            return Vote(self.name, True, quality, f"High quality path: {quality:.3f}")
        return Vote(self.name, False, 1.0 - quality, f"Low quality: {quality:.3f}")
    
    def _compute_quality(self, trajectory: List[np.ndarray], target: np.ndarray) -> float:
        """
        Compute quality from:
        1. Smoothness (low angular variation)
        2. Progress (monotonic approach to target)
        3. Diversity (no loops)
        """
        # 1. Smoothness: average angle between consecutive segments
        angles = []
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(np.arccos(cos_angle))
        
        smoothness = 1.0 - (np.mean(angles) / np.pi if angles else 0.5)
        
        # 2. Progress: fraction of initial distance covered
        initial_dist = np.linalg.norm(trajectory[0] - target)
        final_dist = np.linalg.norm(trajectory[-1] - target)
        progress = (initial_dist - final_dist) / (initial_dist + 1e-9)
        progress = max(0.0, min(1.0, progress))
        
        # 3. Diversity: average distance between consecutive points
        step_sizes = [np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                      for i in range(len(trajectory) - 1)]
        diversity = min(1.0, np.mean(step_sizes) * 10) if step_sizes else 0.5
        
        # Weighted combination
        quality = 0.3 * smoothness + 0.5 * progress + 0.2 * diversity
        return float(quality)


class AdaptiveEarlyStoppingCommittee:
    """
    Adaptive committee that adjusts thresholds based on navigation mode.
    
    - Sprint mode: Tighter thresholds (stop quickly)
    - Creative mode: Looser thresholds (allow more exploration)
    - Balanced: Default thresholds
    """
    
    def __init__(self):
        self.base_convergence_threshold = 0.95
        self.voters = []
        self._init_voters()
    
    def _init_voters(self):
        self.voters = [
            ConvergenceVoter(threshold=self.base_convergence_threshold),
            ProgressVoter(patience=5, min_progress=0.005),
            PathQualityVoter(min_quality=0.75)
        ]
    
    def should_stop(
        self,
        trajectory: List[np.ndarray],
        target: np.ndarray,
        current_step: int,
        context: Dict[str, Any]
    ) -> StopDecision:
        
        # 1. Adapt thresholds based on mode
        mode = context.get('mode', 'balanced')
        self._adapt_thresholds(mode)
        
        # 2. Collect votes
        votes = []
        for voter in self.voters:
            votes.append(voter.vote(trajectory, target, current_step, context))
        
        # 3. Decision logic (majority with weights)
        stop_votes = [v for v in votes if v.should_stop]
        
        # Require at least 2 voters to agree (majority)
        should_stop = len(stop_votes) >= 2
        
        # Or if convergence alone is confident enough
        convergence_vote = next((v for v in votes if v.voter_name == 'convergence'), None)
        if convergence_vote and convergence_vote.should_stop and convergence_vote.confidence > 0.98:
            should_stop = True
        
        agreement_ratio = len(stop_votes) / len(votes) if votes else 0.0
        primary_reason = stop_votes[0].reason if stop_votes else ""
        
        return StopDecision(
            should_stop=should_stop,
            votes=votes,
            agreement_ratio=agreement_ratio,
            primary_reason=primary_reason,
            success_prediction=should_stop and convergence_vote.confidence > 0.9 if convergence_vote else False
        )
    
    def _adapt_thresholds(self, mode):
        """Adjust voter thresholds based on navigation mode."""
        if mode == 'sprint' or (hasattr(mode, 'value') and mode.value == 'sprint'):
            # Sprint: be more aggressive, stop quickly
            self.voters[0].threshold = 0.92  # Convergence
        elif mode == 'creative' or (hasattr(mode, 'value') and mode.value == 'creative'):
            # Creative: be more lenient, allow exploration
            self.voters[0].threshold = 0.98  # Convergence - higher bar
        else:
            # Balanced: default
            self.voters[0].threshold = self.base_convergence_threshold


# Keep legacy name for backward compatibility
EarlyStoppingCommittee = AdaptiveEarlyStoppingCommittee

