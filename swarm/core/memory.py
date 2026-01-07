"""
swarm/memory.py

Persistent Topological Memory.
Consolidated from v3_1/persistent_memory.py.

Solves the problem of memory being discarded between sessions.
Implements:
1. Hierarchical storage (episodic + semantic + danger zones)
2. Importance scoring for retention
3. Cross-session persistence
4. LRU cache for frequently accessed memories
"""

import numpy as np
import json
import time
import os
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

@dataclass
class TrajectoryMemory:
    """A stored trajectory with metadata"""
    memory_id: str
    start_code: str
    target_code: str
    trajectory_length: int
    success: bool
    final_similarity: float
    complexity: float
    agent_contributions: Dict[str, float]
    timestamp: float
    access_count: int
    importance: float
    
    # Compressed trajectory (not full embeddings)
    trajectory_signature: List[float]  # Key points only


@dataclass
class DangerZone:
    """A known problematic region"""
    zone_id: str
    center: List[float]
    radius: float
    failure_count: int
    last_failure: float
    description: str


class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Evict oldest if over capacity
        while len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)

    def values(self):
        return self.cache.values()


class PersistentTopologicalMemory:
    """
    Persistent memory system for Swarm Navigator.
    
    Stores:
    - Successful trajectories for reuse
    - Danger zones to avoid
    - Agent performance statistics
    
    Persists to disk for cross-session memory.
    """
    
    def __init__(self, save_path: str = './data/swarm_memory.json'):
        # Just use folder part
        self.storage_path = Path(os.path.dirname(save_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.trajectory_cache = LRUCache(maxsize=1000)
        self.danger_zones: Dict[str, DangerZone] = {}
        self.agent_stats: Dict[str, Dict[str, float]] = {}
        
        # Load existing data
        self._load_from_disk()
        
        # Session statistics
        self.session_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'trajectories_saved': 0,
            'danger_zones_added': 0,
        }
    
    def _load_from_disk(self):
        """Load persisted data from disk"""
        # Load trajectories
        traj_file = self.storage_path / 'trajectories.json'
        if traj_file.exists():
            try:
                with open(traj_file, 'r') as f:
                    data = json.load(f)
                    for mem_id, mem_data in data.items():
                        memory = TrajectoryMemory(**mem_data)
                        self.trajectory_cache.set(mem_id, memory)
            except Exception as e:
                print(f"Warning: Could not load trajectories: {e}")
        
        # Load danger zones
        danger_file = self.storage_path / 'danger_zones.json'
        if danger_file.exists():
            try:
                with open(danger_file, 'r') as f:
                    data = json.load(f)
                    for zone_id, zone_data in data.items():
                        self.danger_zones[zone_id] = DangerZone(**zone_data)
            except Exception as e:
                print(f"Warning: Could not load danger zones: {e}")
        
        # Load agent stats
        stats_file = self.storage_path / 'agent_stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    self.agent_stats = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load agent stats: {e}")
    
    def save_to_disk(self):
        """Persist data to disk"""
        # Save trajectories (most important ones)
        trajectories = {}
        for key in list(self.trajectory_cache.cache.keys())[-500:]:  # Save top 500
            memory = self.trajectory_cache.cache[key]
            if hasattr(memory, '__dict__'):
                trajectories[key] = asdict(memory)
            else:
                trajectories[key] = memory
        
        traj_file = self.storage_path / 'trajectories.json'
        with open(traj_file, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        # Save danger zones
        danger_file = self.storage_path / 'danger_zones.json'
        with open(danger_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.danger_zones.items()}, f, indent=2)
        
        # Save agent stats
        stats_file = self.storage_path / 'agent_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.agent_stats, f, indent=2)
    
    def save_trajectory(
        self,
        trajectory: List[np.ndarray],
        start: np.ndarray,
        target: np.ndarray,
        success: bool,
        efficiency: float, # mapped to final_similarity logic sometimes, see callsite
        mean_curvature: float,
        neurotypes_used: Dict[str, float],
        complexity: float = 0.5
    ) -> str:
        """
        Save a trajectory to memory.
        """
        # Simple string representation for hashing
        start_text = f"emb_{hash(start.tobytes())}"
        target_text = f"emb_{hash(target.tobytes())}"
        
        # Generate memory ID
        memory_id = f"{hash((start_text, target_text, time.time()))}_{int(time.time())}"
        
        final_similarity = efficiency # Using efficiency as proxy for now
        
        # Calculate importance
        importance = self._calculate_importance(
            success, final_similarity, complexity, trajectory
        )
        
        # Create trajectory signature (compressed)
        signature = self._create_trajectory_signature(trajectory)
        
        memory = TrajectoryMemory(
            memory_id=memory_id,
            start_code=start_text,
            target_code=target_text,
            trajectory_length=len(trajectory),
            success=success,
            final_similarity=final_similarity,
            complexity=complexity,
            agent_contributions=neurotypes_used,
            timestamp=time.time(),
            access_count=1,
            importance=importance,
            trajectory_signature=signature
        )
        
        # Store in cache
        self.trajectory_cache.set(memory_id, memory)
        self.session_stats['trajectories_saved'] += 1
        
        return memory_id
    
    def _calculate_importance(
        self,
        success: bool,
        final_similarity: float,
        complexity: float,
        trajectory: List[np.ndarray]
    ) -> float:
        """Calculate importance score for memory retention"""
        score = 0.0
        
        # Success matters most (40%)
        score += (1.0 if success else 0.0) * 0.4
        
        # Final similarity (30%)
        score += final_similarity * 0.3
        
        # Complexity bonus (20%) - hard problems are valuable
        score += complexity * 0.2
        
        # Efficiency (10%) - shorter paths are better
        efficiency = 1.0 / (1.0 + len(trajectory) / 50.0)
        score += efficiency * 0.1
        
        return min(score, 1.0)
    
    def _create_trajectory_signature(
        self,
        trajectory: List[np.ndarray],
        n_points: int = 5
    ) -> List[float]:
        """Create compressed signature from trajectory"""
        if not trajectory:
            return []
        
        # Take evenly spaced points
        indices = np.linspace(0, len(trajectory) - 1, n_points, dtype=int)
        
        signature = []
        for i in indices:
            # Use first 10 dimensions as signature
            point = trajectory[i][:10] if len(trajectory[i]) >= 10 else trajectory[i]
            signature.extend(point.tolist())
        
        return signature
    
    def find_similar_trajectories(
        self,
        start_emb: np.ndarray,
        target_emb: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.6
    ) -> List[TrajectoryMemory]:
        """
        Find past trajectories with similar start/target embeddings.
        
        Uses trajectory signatures (first 10 dims of key points) to match.
        Returns top_k most relevant trajectories sorted by importance.
        
        Args:
            start_emb: Current navigation start embedding
            target_emb: Current navigation target embedding  
            top_k: Maximum trajectories to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of TrajectoryMemory objects from similar past navigations
        """
        if not self.trajectory_cache.cache:
            self.session_stats['cache_misses'] += 1
            return []
        
        # Create signature from current start/target
        query_sig = np.concatenate([
            start_emb[:10] if len(start_emb) >= 10 else start_emb,
            target_emb[:10] if len(target_emb) >= 10 else target_emb
        ])
        query_norm = np.linalg.norm(query_sig)
        if query_norm > 1e-9:
            query_sig = query_sig / query_norm
        
        candidates = []
        
        for mem_id, memory in self.trajectory_cache.cache.items():
            if not hasattr(memory, 'trajectory_signature') or not memory.trajectory_signature:
                continue
            
            # Compare first 20 elements of signature (start + target approximation)
            sig = np.array(memory.trajectory_signature[:20])
            if len(sig) < 20:
                continue
                
            sig_norm = np.linalg.norm(sig)
            if sig_norm > 1e-9:
                sig = sig / sig_norm
            
            # Cosine similarity
            similarity = float(np.dot(query_sig[:len(sig)], sig))
            
            if similarity >= min_similarity:
                candidates.append((memory, similarity))
        
        if not candidates:
            self.session_stats['cache_misses'] += 1
            return []
        
        self.session_stats['cache_hits'] += 1
        
        # Sort by (similarity * importance) and return top_k
        candidates.sort(key=lambda x: x[1] * x[0].importance, reverse=True)
        
        # Update access count for returned memories
        results = []
        for memory, _ in candidates[:top_k]:
            memory.access_count += 1
            results.append(memory)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        total = len(self.trajectory_cache)
        successes = sum(1 for t in self.trajectory_cache.values() if t.success)
        
        return {
            'total_trajectories': total,
            'success_rate': successes / total if total > 0 else 0.0,
            'total_danger_zones': len(self.danger_zones),
            'session_stats': self.session_stats,
            'cache_hit_rate': (
                self.session_stats['cache_hits'] / 
                max(1, self.session_stats['cache_hits'] + self.session_stats['cache_misses'])
            ),
        }
    
    def __del__(self):
        """Save on destruction"""
        try:
            self.save_to_disk()
        except:
            pass
