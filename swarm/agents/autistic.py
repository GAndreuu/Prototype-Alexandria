"""
swarm/agents/autistic.py

AutisticAgent - Hyperfocus with delayed integration.

This agent operates with intense local focus and pattern detection,
but delays global integration. It builds understanding from details
upward, excelling at finding patterns others miss.

Inspiration: Autistic cognition research, detail-focused processing,
             weak central coherence, pattern recognition strengths.
"""

import numpy as np
from typing import Optional, List, Deque
from collections import deque

from .base import BaseAgent
from ..core import (
    NavigationStep, 
    HeuristicPersonality,
    NeurodiverseProfile, 
    TopologicalState, 
    NeurotypeName,
    Context,
    AUTISTIC_PROFILE
)


class AutisticAgent(BaseAgent):
    """
    Hyperfocus agent with delayed integration.
    """
    
    def __init__(self, profile: Optional[NeurodiverseProfile] = None):
        self.profile = profile or AUTISTIC_PROFILE
        
        # Delayed integration buffer
        self.integration_buffer: Deque[np.ndarray] = deque(maxlen=10)
        
        # Pattern memory
        self.local_patterns: List[np.ndarray] = []
        self.pattern_weights: List[float] = []
        
        # Focus anchor
        self.focus_point: Optional[np.ndarray] = None
        self.focus_strength = 0.0
        
    def propose(self, ctx: Context) -> NavigationStep:
        """
        Propose a focused, pattern-aware step.
        """
        # Direct vector (still needed as reference)
        direct = ctx.target_emb - ctx.current
        direct_norm = np.linalg.norm(direct)
        
        if direct_norm > 1e-8:
            direct = direct / direct_norm
        
        # Update integration buffer
        self.integration_buffer.append(ctx.current.copy())
        
        # Check if we should integrate (delayed)
        if len(self.integration_buffer) >= self.profile.integration_delay:
            integrated = self._integrate_buffer()
        else:
            integrated = None
        
        # Detect local patterns
        pattern_direction = self._detect_local_pattern(ctx)
        
        # Maintain or update focus
        self._update_focus(ctx)
        
        # Compute direction
        direction = np.zeros_like(ctx.current)
        total_weight = 0.0
        
        # Direct component (reduced weight due to delayed integration)
        direct_weight = 0.3 * (1.0 - self.profile.lateral_inhibition)
        direction += direct_weight * direct
        total_weight += direct_weight
        
        # Pattern component
        if pattern_direction is not None:
            pattern_weight = 0.4
            direction += pattern_weight * pattern_direction
            total_weight += pattern_weight
        
        # Focus component
        if self.focus_point is not None and self.focus_strength > 0.2:
            focus_dir = self.focus_point - ctx.current
            focus_norm = np.linalg.norm(focus_dir)
            if focus_norm > 1e-8:
                focus_dir = focus_dir / focus_norm
                focus_weight = 0.3 * self.focus_strength
                direction += focus_weight * focus_dir
                total_weight += focus_weight
        
        # Integrated direction (if available)
        if integrated is not None:
            int_weight = 0.2
            direction += int_weight * integrated
            total_weight += int_weight
        
        # Normalize
        if total_weight > 1e-8:
            direction = direction / total_weight
        else:
            direction = direct
            
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-8:
            direction = direction / dir_norm
        
        # High confidence when patterns are detected
        pattern_detected = pattern_direction is not None
        base_confidence = 0.5
        confidence = base_confidence + 0.3 * (1 if pattern_detected else 0)
        confidence *= self.profile.coherence
        
        # --- Topological Awareness ---
        topo_state = self._compute_autistic_topology(ctx)
        
        return NavigationStep(
            agent_id="autistic",
            neurotype=NeurotypeName.AUTISTIC,
            direction=direction,
            confidence=confidence,
            reasoning=f"Autistic mode (focus={self.focus_strength:.2f}, patterns={len(self.local_patterns)})",
            personality=HeuristicPersonality.INTERPOLATION,
            topological_state=topo_state
        )
    
    def _integrate_buffer(self) -> Optional[np.ndarray]:
        """Integrate delayed observations into a coherent direction."""
        if len(self.integration_buffer) < 2:
            return None
        
        positions = list(self.integration_buffer)
        
        # Weighted average favoring recent
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights = weights / weights.sum()
        
        integrated = np.zeros_like(positions[0])
        for i in range(1, len(positions)):
            movement = positions[i] - positions[i-1]
            integrated += weights[i] * movement
        
        norm = np.linalg.norm(integrated)
        if norm > 1e-8:
            return integrated / norm
        return None
    
    def _detect_local_pattern(self, ctx: Context) -> Optional[np.ndarray]:
        """Detect repeating patterns in local trajectory."""
        if len(ctx.history) < 4:
            return None
        
        recent = ctx.history[-4:]
        directions = []
        for i in range(len(recent) - 1):
            d = recent[i+1] - recent[i]
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                directions.append(d / norm)
        
        if len(directions) < 2:
            return None
        
        similarities = []
        for i in range(len(directions) - 1):
            sim = float(np.dot(directions[i], directions[i+1]))
            similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        
        if abs(avg_sim) > 0.7:
            if avg_sim > 0.7:
                return directions[-1]
            else:
                base = directions[-1]
                perp = np.random.randn(len(base))
                perp = perp - np.dot(perp, base) * base
                norm = np.linalg.norm(perp)
                if norm > 1e-8:
                    self.local_patterns.append(perp / norm)
                    if len(self.local_patterns) > 5:
                        self.local_patterns.pop(0)
                    return perp / norm
        return None
    
    def _update_focus(self, ctx: Context):
        """Update the focus point based on local analysis."""
        if self.focus_point is None:
            if len(ctx.history) > 2:
                self.focus_point = ctx.history[len(ctx.history)//2].copy()
                self.focus_strength = 0.5
        else:
            self.focus_strength *= 0.95
            if len(self.local_patterns) > 0:
                self.focus_point = ctx.current.copy()
                self.focus_strength = min(1.0, self.focus_strength + 0.2)
    
    def _compute_autistic_topology(self, ctx: Context) -> TopologicalState:
        """Compute topology from autistic perspective."""
        dist = np.linalg.norm(ctx.target_emb - ctx.current)
        energy = dist / (ctx.initial_dist + 1e-8)
        
        if len(ctx.history) >= 2:
            prev_dir = ctx.current - ctx.history[-1]
            if len(ctx.history) >= 3:
                prev_prev_dir = ctx.history[-1] - ctx.history[-2]
                cos_angle = np.dot(prev_dir, prev_prev_dir) / (
                    np.linalg.norm(prev_dir) * np.linalg.norm(prev_prev_dir) + 1e-8
                )
                curvature = (1.0 - cos_angle) * 1.2
            else:
                curvature = 0.3
        else:
            curvature = 0.0
        
        return TopologicalState(
            position=ctx.current.copy(),
            energy=energy,
            curvature=min(1.0, curvature),
            density=0.5,
            near_collapse=curvature > 0.9,
            entropy=0.3 + 0.2 * curvature
        )
    
    def reset(self):
        """Reset agent state."""
        self.integration_buffer.clear()
        self.local_patterns = []
        self.pattern_weights = []
        self.focus_point = None
        self.focus_strength = 0.0
