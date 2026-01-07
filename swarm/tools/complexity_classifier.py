"""
swarm/tools/complexity_classifier.py

Complexity Classifier + Sprint Mode
Consolidated from v3_1/complexity_classifier.py.

Detects when navigation path is simple enough for "Sprint Mode":
- High direct similarity (>0.85)
- Low local curvature (<0.1)
- Known short path in Mycelial graph
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from ..core import NavigationMode, ModeConfig

class ComplexityClassifier:
    """
    Classifies navigation complexity to select optimal mode.
    
    Key insight: Most navigations are simple and don't need
    the full neurodiverse swarm. Detecting this early saves
    significant compute.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Thresholds for Sprint mode
        self.sprint_thresholds = {
            'similarity': self.config.get('sprint_similarity', 0.70),  # Lowered from 0.85
            'curvature': self.config.get('sprint_curvature', 0.1),
            'mycelial_hops': self.config.get('sprint_mycelial_hops', 2),
        }
        
        # Thresholds for Cautious mode
        self.cautious_thresholds = {
            'curvature': self.config.get('cautious_curvature', 0.5),
            'singularity_risk': self.config.get('cautious_singularity', 0.7),
        }
        
        # Mode configurations
        self.mode_configs = {
            NavigationMode.SPRINT: ModeConfig(
                max_steps=15,
                min_confidence=0.95, # reusing min_confidence as threshold proxy for config
                neurotype_mix={'collapse': 0.8, 'autistic': 0.2},
                early_stopping_patience=2
            ),
            NavigationMode.BALANCED: ModeConfig(
                max_steps=30,
                min_confidence=0.90,
                neurotype_mix={'balanced': 0.4, 'critical': 0.2, 'psych': 0.2, 'autistic': 0.2},
                early_stopping_patience=5
            ),
            NavigationMode.CAUTIOUS: ModeConfig(
                max_steps=40,
                min_confidence=0.92,
                neurotype_mix={'critical': 0.5, 'autistic': 0.3, 'relaxed': 0.2},
                early_stopping_patience=7
            ),
            NavigationMode.CREATIVE: ModeConfig(
                max_steps=50,
                min_confidence=0.85,
                neurotype_mix={'psych': 0.4, 'critical': 0.3, 'relaxed': 0.3},
                early_stopping_patience=10
            ),
        }
        
        # Statistics for learning
        self.classification_stats = {
            'total': 0,
            'sprint': 0,
            'balanced': 0,
            'cautious': 0,
            'creative': 0,
        }
    
    def classify(
        self, 
        start_emb: np.ndarray, 
        target_emb: np.ndarray,
        context: Optional[Dict] = None
    ) -> Tuple[NavigationMode, ModeConfig]:
        """
        Classify navigation complexity and return optimal mode.
        """
        context = context or {}
        
        # 1. Check for Sprint conditions
        if self._should_sprint(start_emb, target_emb, context):
            self._update_stats('sprint')
            return NavigationMode.SPRINT, self.mode_configs[NavigationMode.SPRINT]
        
        # 2. Check for Cautious conditions
        if self._should_be_cautious(start_emb, target_emb, context):
            self._update_stats('cautious')
            return NavigationMode.CAUTIOUS, self.mode_configs[NavigationMode.CAUTIOUS]
        
        # 3. Check for Creative conditions
        if self._should_be_creative(start_emb, target_emb, context):
            self._update_stats('creative')
            return NavigationMode.CREATIVE, self.mode_configs[NavigationMode.CREATIVE]
        
        # 4. Default to Balanced
        self._update_stats('balanced')
        return NavigationMode.BALANCED, self.mode_configs[NavigationMode.BALANCED]
    
    def _should_sprint(
        self, 
        start_emb: np.ndarray, 
        target_emb: np.ndarray, 
        context: Dict
    ) -> bool:
        """Determines if Sprint mode is appropriate."""
        # 1. High similarity check
        similarity = self._cosine_similarity(start_emb, target_emb)
        if similarity > self.sprint_thresholds['similarity']:
            return True
        
        # 2. Low curvature check (if topology available)
        topology_analyzer = context.get('topology_analyzer')
        if topology_analyzer:
            try:
                curvature = topology_analyzer.estimate_curvature_between(
                    start_emb, target_emb
                )
                if curvature < self.sprint_thresholds['curvature']:
                    return True
            except:
                pass
        
        # 3. Mycelial short path check
        mycelial = context.get('mycelial')
        if mycelial and hasattr(mycelial, 'find_shortest_path'):
            try:
                path = mycelial.find_shortest_path(start_emb, target_emb)
                if path and len(path) <= self.sprint_thresholds['mycelial_hops']:
                    return True
            except:
                pass
        
        # 4. Medium-high similarity with no obstacles
        if similarity > 0.6 and not self._has_obstacles(start_emb, target_emb, context):
            return True
        
        return False
    
    def _should_be_cautious(
        self, 
        start_emb: np.ndarray, 
        target_emb: np.ndarray, 
        context: Dict
    ) -> bool:
        """Determines if Cautious mode is appropriate."""
        topology_analyzer = context.get('topology_analyzer')
        if not topology_analyzer:
            return False
        
        try:
            # Check curvature
            curvature = topology_analyzer.estimate_curvature_between(
                start_emb, target_emb
            )
            if curvature > self.cautious_thresholds['curvature']:
                return True
            
            # Check singularity risk - Removed for consolidation simplicity or assumed handled in topo logic
            # singularity_info = topology_analyzer.detect_singularity_risk(start_emb)
            # if singularity_info.get('risk', 0) > self.cautious_thresholds['singularity_risk']:
            #     return True
                
        except:
            pass
        
        # Check memory for previous failures
        memory = context.get('memory')
        if memory and hasattr(memory, 'get_failure_rate'):
            try:
                failure_rate = memory.get_failure_rate_near(start_emb, radius=0.3)
                if failure_rate > 0.5:
                    return True
            except:
                pass
        
        return False
    
    def _should_be_creative(
        self, 
        start_emb: np.ndarray, 
        target_emb: np.ndarray, 
        context: Dict
    ) -> bool:
        """Determines if Creative mode is appropriate."""
        similarity = self._cosine_similarity(start_emb, target_emb)
        
        # Very low similarity = need creativity
        if similarity < 0.3:
            return True
        
        # No mycelial path = need to explore
        mycelial = context.get('mycelial')
        if mycelial and hasattr(mycelial, 'find_shortest_path'):
            try:
                path = mycelial.find_shortest_path(start_emb, target_emb)
                if not path or len(path) > 10:
                    return True
            except:
                pass
        
        return False
    
    def _has_obstacles(
        self, 
        start_emb: np.ndarray, 
        target_emb: np.ndarray, 
        context: Dict
    ) -> bool:
        """Check if there are known obstacles between start and target"""
        memory = context.get('memory')
        if not memory:
            return False
        
        try:
            # Sample points along the line
            for t in [0.25, 0.5, 0.75]:
                point = start_emb + t * (target_emb - start_emb)
                if memory.is_danger_zone(point):
                    return True
        except:
            pass
        
        return False
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _update_stats(self, mode_name: str):
        """Update classification statistics"""
        self.classification_stats['total'] += 1
        self.classification_stats[mode_name] = self.classification_stats.get(mode_name, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return self.classification_stats

