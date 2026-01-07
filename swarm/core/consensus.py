"""
swarm/consensus.py

Fair Consensus System.
Consolidated from v3_1/fair_consensus.py.

Solves the problem of "Balanced" agent dominating consensus (~60%).
Implements:
1. Veto limits - No agent can exceed 30% weight
2. Rotation - Underused neurodiverse agents get boosted periodically
3. Error penalties - Agents that propose bad directions lose weight
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

from .types import (
    NavigationStep, Context, NeurodiverseProfile, 
    NeurotypeName, ConsensusResult
)

class AgentCategory(Enum):
    """Agent categories for balancing"""
    CLASSIC = "classic"  # Direct, Gradient, Interpolation
    NEURODIVERSE = "neurodiverse"  # Psych, Critical, Autistic, Collapse
    MYCELIAL = "mycelial"  # MycelialBridge

@dataclass
class WeightedProposal:
    """A proposal with calculated weight"""
    agent_id: str
    agent_type: str
    direction: np.ndarray
    confidence: float
    original_weight: float
    adjusted_weight: float
    reasoning: str

class NeurodiverseConsensus:
    """
    Fair consensus system with veto limits and agent rotation.
    
    Key principles:
    1. No single agent dominates (max 30% weight)
    2. Neurodiverse agents collectively have minimum 40% weight
    3. Rotation promotes underused agents every N steps
    4. Error tracking penalizes consistently wrong agents
    """
    
    def __init__(self, config: Optional[Dict] = None, memory=None):
        self.config = config or {}
        self.memory = memory  # PersistentTopologicalMemory for feedback loop
        
        # Veto limits
        self.max_individual_weight = self.config.get('max_individual_weight', 0.30)
        self.min_neurodiverse_weight = self.config.get('min_neurodiverse_weight', 0.40)
        
        # Rotation settings
        self.rotation_interval = self.config.get('rotation_interval', 10)
        self.boost_factor = self.config.get('boost_factor', 1.5)
        
        # Error penalty settings
        self.error_decay = self.config.get('error_decay', 0.9)  # How fast errors are forgotten
        self.error_penalty_factor = self.config.get('error_penalty_factor', 0.3)
        
        # Agent categorization
        self.agent_categories = {
            'direct': AgentCategory.CLASSIC,
            'gradient': AgentCategory.CLASSIC,
            'interpolation': AgentCategory.CLASSIC,
            'momentum': AgentCategory.CLASSIC,
            'balanced': AgentCategory.CLASSIC,
            'collapse': AgentCategory.NEURODIVERSE,
            'critical': AgentCategory.NEURODIVERSE,
            'psych': AgentCategory.NEURODIVERSE,
            'psychedelic': AgentCategory.NEURODIVERSE,
            'autistic': AgentCategory.NEURODIVERSE,
            'relaxed': AgentCategory.NEURODIVERSE,
            'mycelial_bridge': AgentCategory.MYCELIAL,
            'explorer': AgentCategory.CLASSIC,
        }
        
        # Historical performance tracking
        self.agent_contributions = defaultdict(float)
        self.agent_errors = defaultdict(float)
        self.agent_successes = defaultdict(float)
        
        # Rotation state
        self.current_step = 0
        self.rotation_schedule = []
        self.current_boosted = None
    
    def compute_consensus(
        self,
        proposals: List[NavigationStep], 
        ctx: Any, # Context object
        target_mix: Optional[Dict[NeurotypeName, float]] = None
    ) -> ConsensusResult:
        """
        Compute consensus direction from multiple agents.
        (Wrapper around calculate_fair_weights for API compatibility)
        
        FEEDBACK LOOP: Consults memory BEFORE voting to adjust weights
        based on what worked in similar past navigations.
        """
        # FEEDBACK LOOP: Consult memory before voting
        if self.memory and hasattr(ctx, 'start_emb') and hasattr(ctx, 'target_emb'):
            try:
                similar = self.memory.find_similar_trajectories(
                    ctx.start_emb, ctx.target_emb
                )
                if similar:
                    target_mix = self._adjust_mix_from_history(similar, target_mix)
            except Exception:
                pass  # Graceful degradation if memory query fails
        
        if not proposals:
            # Fallback
            return ConsensusResult(
                direction=np.zeros(384), # Should be random or fail
                confidence=0.0,
                reasoning="No proposals",
                contributing_neurotypes={}
            )

        # Convert simple list of proposals to format for calculation if needed,
        # but here we can just pass them directly as we adapt the method below.
        
        # 1. Calculate weights
        weighted_proposals = self.calculate_fair_weights(
            proposals, ctx.step
        )
        
        # 2. Combine
        final_direction = np.zeros_like(proposals[0].direction)
        for p in weighted_proposals:
            final_direction += p.adjusted_weight * p.direction
            
        # Normalize
        norm = np.linalg.norm(final_direction)
        if norm > 1e-9:
            final_direction = final_direction / norm
            
        # Confidence
        avg_confidence = sum(p.confidence * p.adjusted_weight for p in weighted_proposals)
        
        # Reasoning
        top = sorted(weighted_proposals, key=lambda x: x.adjusted_weight, reverse=True)[:3]
        reason_parts = [f"{p.agent_id}:{p.adjusted_weight:.2f}" for p in top]
        reasoning = f"Consensus({', '.join(reason_parts)})"
        
        # Tracking for result
        contribs = {}
        for p in weighted_proposals:
            # Map agent_type back to NeurotypeName if possible, or string
            # For simplicity using string keys in this dict, or map if strict
            map_name = self._map_to_neurotype(p.agent_type)
            contribs[map_name] = contribs.get(map_name, 0) + p.adjusted_weight

        return ConsensusResult(
            direction=final_direction,
            confidence=avg_confidence,
            reasoning=reasoning,
            contributing_neurotypes=contribs
        )

    def calculate_fair_weights(
        self,
        proposals: List[NavigationStep],
        step: int
    ) -> List[WeightedProposal]:
        """
        Calculate fair weights for all proposals.
        """
        self.current_step = step
        
        if not proposals:
            return []
        
        # 1. Calculate initial weights from confidence
        weighted = []
        for p in proposals:
            agent_id = p.agent_id or 'unknown'
            agent_type = self._get_agent_type(agent_id)
            confidence = p.confidence
            
            weighted.append(WeightedProposal(
                agent_id=agent_id,
                agent_type=agent_type,
                direction=p.direction,
                confidence=confidence,
                original_weight=confidence,
                adjusted_weight=confidence,  # Will be modified
                reasoning=p.reasoning
            ))
        
        # 2. Apply veto limits
        weighted = self._apply_veto_limits(weighted)
        
        # 3. Ensure minimum neurodiverse weight
        weighted = self._ensure_neurodiverse_minimum(weighted)
        
        # 4. Apply rotation boost
        if step % self.rotation_interval == 0:
            weighted = self._apply_rotation_boost(weighted)
        
        # 5. Apply error penalties
        weighted = self._apply_error_penalties(weighted)
        
        # 6. Normalize weights
        weighted = self._normalize_weights(weighted)
        
        # 7. Track contributions
        self._track_contributions(weighted)
        
        return weighted
    
    def _map_to_neurotype(self, agent_type: str) -> NeurotypeName:
        """Map string agent type to NeurotypeName Enum if possible."""
        try:
            return NeurotypeName(agent_type)
        except ValueError:
            # Fallback mapping
            mapping = {
                'collapse': NeurotypeName.COLLAPSE,
                'critical': NeurotypeName.CRITICAL,
                'psych': NeurotypeName.PSYCH,
                'psychedelic': NeurotypeName.PSYCH,
                'autistic': NeurotypeName.AUTISTIC,
                'balanced': NeurotypeName.BALANCED,
                'relaxed': NeurotypeName.RELAXED
            }
            return mapping.get(agent_type, NeurotypeName.BALANCED)

    def _adjust_mix_from_history(
        self,
        similar_trajectories: list,
        current_mix: Optional[Dict[NeurotypeName, float]] = None
    ) -> Dict[NeurotypeName, float]:
        """
        Adjust neurotype weights based on what worked in similar past trajectories.
        
        Strategy: Incremental boost
        - Successful trajectories contribute their agent_contributions
        - Weights are boosted proportionally to historical success
        - Original mix is preserved but enhanced
        
        Args:
            similar_trajectories: List of TrajectoryMemory from find_similar_trajectories
            current_mix: Existing target mix (if any)
            
        Returns:
            Adjusted target mix with historical boosts applied
        """
        if not similar_trajectories:
            return current_mix
        
        # Aggregate contributions from successful trajectories
        historical_contributions: Dict[str, float] = {}
        total_importance = 0.0
        
        # Current time for decay calculation
        import time
        now = time.time()
        DECAY_RATE = 0.95  # 5% decay per day
        SECONDS_PER_DAY = 86400
        
        for memory in similar_trajectories:
            if not memory.success:
                continue  # Only learn from successes
            
            # Temporal decay: recent trajectories weight more
            age_days = (now - memory.timestamp) / SECONDS_PER_DAY
            decay = DECAY_RATE ** age_days
            
            importance = memory.importance * decay
            total_importance += importance
            
            for agent_type, contribution in memory.agent_contributions.items():
                if agent_type not in historical_contributions:
                    historical_contributions[agent_type] = 0.0
                historical_contributions[agent_type] += contribution * importance
        
        if total_importance == 0 or not historical_contributions:
            return current_mix
        
        # Normalize
        for agent_type in historical_contributions:
            historical_contributions[agent_type] /= total_importance
        
        # Create or enhance mix
        if current_mix is None:
            current_mix = {}
        
        # Apply boost (30% influence from history, 70% from current)
        HISTORY_WEIGHT = 0.3
        result_mix = dict(current_mix)
        
        for agent_type, hist_weight in historical_contributions.items():
            neurotype = self._map_to_neurotype(agent_type)
            
            current_value = result_mix.get(neurotype, 0.0)
            boosted_value = current_value * (1 - HISTORY_WEIGHT) + hist_weight * HISTORY_WEIGHT
            result_mix[neurotype] = boosted_value
        
        return result_mix

    def _get_agent_type(self, agent_id: str) -> str:
        """Get agent type from ID"""
        # Handle agent IDs like "collapse_agent", "critical", etc.
        agent_id_lower = agent_id.lower().replace('_agent', '')
        return agent_id_lower
    
    def _apply_veto_limits(
        self, 
        weighted: List[WeightedProposal]
    ) -> List[WeightedProposal]:
        """Apply maximum weight limits per agent"""
        total_weight = sum(p.adjusted_weight for p in weighted)
        if total_weight == 0:
            return weighted
        
        # Calculate excess weight to redistribute
        excess_total = 0.0
        
        for p in weighted:
            proportion = p.adjusted_weight / total_weight
            if proportion > self.max_individual_weight:
                # Calculate excess
                max_allowed = self.max_individual_weight * total_weight
                excess = p.adjusted_weight - max_allowed
                excess_total += excess
                p.adjusted_weight = max_allowed
        
        # Redistribute excess to under-limit agents
        under_limit = [
            p for p in weighted 
            if p.adjusted_weight / total_weight < self.max_individual_weight
        ]
        
        if under_limit and excess_total > 0:
            boost_per_agent = excess_total / len(under_limit)
            for p in under_limit:
                p.adjusted_weight += boost_per_agent
        
        return weighted
    
    def _ensure_neurodiverse_minimum(
        self,
        weighted: List[WeightedProposal]
    ) -> List[WeightedProposal]:
        """Ensure neurodiverse agents have minimum collective weight"""
        total_weight = sum(p.adjusted_weight for p in weighted)
        if total_weight == 0:
            return weighted
        
        # Calculate neurodiverse proportion
        neurodiverse = [
            p for p in weighted 
            if self.agent_categories.get(p.agent_type) == AgentCategory.NEURODIVERSE
        ]
        
        classic = [
            p for p in weighted 
            if self.agent_categories.get(p.agent_type) == AgentCategory.CLASSIC
        ]
        
        if not neurodiverse:
            return weighted
        
        neurodiverse_weight = sum(p.adjusted_weight for p in neurodiverse)
        neurodiverse_proportion = neurodiverse_weight / total_weight
        
        if neurodiverse_proportion < self.min_neurodiverse_weight:
            # Need to boost neurodiverse
            target_neurodiverse = self.min_neurodiverse_weight * total_weight
            boost_needed = target_neurodiverse - neurodiverse_weight
            
            # Take from classic agents
            if classic:
                reduction_per_classic = boost_needed / len(classic)
                for p in classic:
                    p.adjusted_weight = max(0.01, p.adjusted_weight - reduction_per_classic)
                
                # Add to neurodiverse equally
                boost_per_neuro = boost_needed / len(neurodiverse)
                for p in neurodiverse:
                    p.adjusted_weight += boost_per_neuro
        
        return weighted
    
    def _apply_rotation_boost(
        self,
        weighted: List[WeightedProposal]
    ) -> List[WeightedProposal]:
        """Boost underused neurodiverse agents on rotation schedule"""
        # Find most underused neurodiverse agent
        neurodiverse = [
            p for p in weighted 
            if self.agent_categories.get(p.agent_type) == AgentCategory.NEURODIVERSE
        ]
        
        if not neurodiverse:
            return weighted
        
        # Sort by historical contribution (ascending)
        sorted_neuro = sorted(
            neurodiverse,
            key=lambda p: self.agent_contributions.get(p.agent_type, 0)
        )
        
        # Boost the least used
        if sorted_neuro:
            to_boost = sorted_neuro[0]
            to_boost.adjusted_weight *= self.boost_factor
            self.current_boosted = to_boost.agent_type
        
        return weighted
    
    def _apply_error_penalties(
        self,
        weighted: List[WeightedProposal]
    ) -> List[WeightedProposal]:
        """Penalize agents with high error rates"""
        for p in weighted:
            error_rate = self.agent_errors.get(p.agent_type, 0)
            if error_rate > 0.1:  # More than 10% error rate
                penalty = 1.0 - (error_rate * self.error_penalty_factor)
                p.adjusted_weight *= max(0.1, penalty)
        
        return weighted
    
    def _normalize_weights(
        self,
        weighted: List[WeightedProposal]
    ) -> List[WeightedProposal]:
        """Normalize weights to sum to 1.0"""
        total = sum(p.adjusted_weight for p in weighted)
        if total > 0:
            for p in weighted:
                p.adjusted_weight /= total
        return weighted
    
    def _track_contributions(self, weighted: List[WeightedProposal]):
        """Track historical contributions"""
        for p in weighted:
            self.agent_contributions[p.agent_type] += p.adjusted_weight
    
    def report_error(self, agent_type: str, severity: float = 1.0):
        """Report that an agent made an error"""
        # Decay old errors
        for agent in self.agent_errors:
            self.agent_errors[agent] *= self.error_decay
        
        # Add new error
        self.agent_errors[agent_type] += severity * 0.1
    
    def report_success(self, agent_type: str):
        """Report that an agent contributed to success"""
        self.agent_successes[agent_type] += 1
        # Reduce error rate on success
        self.agent_errors[agent_type] *= 0.8
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        total_contrib = sum(self.agent_contributions.values())
        if total_contrib == 0:
            return {'total_contributions': 0}
        
        # Calculate proportions
        proportions = {
            agent: contrib / total_contrib 
            for agent, contrib in self.agent_contributions.items()
        }
        
        # Group by category
        neurodiverse_proportion = sum(
            p for agent, p in proportions.items()
            if self.agent_categories.get(agent) == AgentCategory.NEURODIVERSE
        )
        
        classic_proportion = sum(
            p for agent, p in proportions.items()
            if self.agent_categories.get(agent) == AgentCategory.CLASSIC
        )
        
        return {
            'total_steps': self.current_step,
            'agent_proportions': proportions,
            'neurodiverse_proportion': neurodiverse_proportion,
            'classic_proportion': classic_proportion,
            'agent_errors': dict(self.agent_errors),
            'current_boosted': self.current_boosted,
        }
    
    def reset(self):
        """Reset for new navigation session"""
        self.current_step = 0
        self.agent_contributions.clear()
        self.agent_errors.clear()
        self.agent_successes.clear()
        self.current_boosted = None
