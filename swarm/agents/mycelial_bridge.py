"""
swarm/agents/mycelial_bridge.py

MycelialBridgeAgent - Finds conceptual bridges via Hebbian graph.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import deque

from .base import BaseAgent
from ..core import Context, NavigationStep, HeuristicPersonality


class MycelialBridgeAgent(BaseAgent):
    """
    Finds conceptual bridges via the Mycelial Hebbian graph.
    
    Replaces expensive geodesic calculations with BFS on discrete graph.
    Complexity: O(V+E) vs O(n³) for geodesics.
    
    FEEDBACK LOOP: Loads and prioritizes previously validated conceptual bridges.
    """
    
    def __init__(self, mycelial, vqvae=None):
        """
        Args:
            mycelial: MycelialReasoningLite instance
            vqvae: VQ-VAE model for encode/decode (optional)
        """
        self.mycelial = mycelial
        self.vqvae = vqvae
        self.bridge_vectors = None
        self._bridge_cache = {}
        self._code_cache = {}
        self._bridge_progress = {}
        self._using_real_codes = False
        self._using_real_decode = False
        
        # FEEDBACK LOOP: Validated bridges from disk
        self._validated_bridges: Dict[tuple, Dict] = {}
        self._load_validated_bridges()
    
    def _load_validated_bridges(self):
        """
        Load validated conceptual bridges from disk.
        
        FEEDBACK LOOP: Previously discovered and validated bridges
        are loaded for priority use in future navigations.
        """
        bridge_path = "data/conceptual_bridges.json"
        
        if not os.path.exists(bridge_path):
            return
        
        try:
            with open(bridge_path, 'r') as f:
                bridges = json.load(f)
            
            for bridge in bridges:
                # Create key from start/target vectors (first 8 dims for efficiency)
                start_vec = bridge.get('start_vector', [])[:8]
                target_vec = bridge.get('target_vector', [])[:8]
                
                if not start_vec or not target_vec:
                    continue
                
                key = (tuple(round(v, 4) for v in start_vec), 
                       tuple(round(v, 4) for v in target_vec))
                
                self._validated_bridges[key] = {
                    'midpoint': np.array(bridge.get('midpoint_vector', [])),
                    'steps': bridge.get('steps', 0),
                    'init_similarity': bridge.get('init_similarity', 0),
                    'timestamp': bridge.get('timestamp', '')
                }
                
        except (json.JSONDecodeError, IOError) as e:
            pass  # Graceful degradation if file is corrupted
    
    def _find_validated_bridge(self, start_emb: np.ndarray, target_emb: np.ndarray) -> Optional[np.ndarray]:
        """
        Check if a validated bridge exists for this start/target pair.
        Returns the midpoint vector if found, None otherwise.
        """
        if not self._validated_bridges:
            return None
        
        # Create lookup key
        start_key = tuple(round(v, 4) for v in start_emb[:8])
        target_key = tuple(round(v, 4) for v in target_emb[:8])
        key = (start_key, target_key)
        
        if key in self._validated_bridges:
            bridge = self._validated_bridges[key]
            if len(bridge['midpoint']) > 0:
                return bridge['midpoint']
        
        return None

    def find_path_between_codes(
        self, 
        start_codes: List[int], 
        target_codes: List[int], 
        max_hops: int = 3
    ) -> List[Tuple[int, int]]:
        """
        BFS on Hebbian graph to find path between code sets.
        
        Args:
            start_codes: VQ-VAE codes of start concept [h0, h1, h2, h3]
            target_codes: VQ-VAE codes of target concept
            max_hops: Maximum search depth
        
        Returns:
            List of nodes (head, code) forming the path
        """
        start_nodes: Set[Tuple[int, int]] = set(
            (h, int(c)) for h, c in enumerate(start_codes) if h < 4
        )
        target_nodes: Set[Tuple[int, int]] = set(
            (h, int(c)) for h, c in enumerate(target_codes) if h < 4
        )
        
        queue = deque([(node, [node]) for node in start_nodes])
        visited = set(start_nodes)
        
        while queue:
            current, path = queue.popleft()
            
            if current in target_nodes:
                return path
            
            if (len(path) - 1) >= max_hops:
                continue
            
            if current in self.mycelial.graph:
                for neighbor, weight in self.mycelial.graph[current].items():
                    if neighbor not in visited and weight > 0.1:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def _get_codes_from_embedding(self, emb: np.ndarray) -> List[int]:
        """
        Convert 384D embedding to VQ-VAE codes.
        
        Uses real quantization if VQ-VAE available, else heuristic hash.
        """
        emb_key = tuple(emb[:8].round(4))
        if emb_key in self._code_cache:
            return self._code_cache[emb_key]
        
        if self.vqvae is not None:
            try:
                import torch
                with torch.no_grad():
                    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                    out = self.vqvae(x)
                if isinstance(out, dict) and 'indices' in out:
                    codes = out['indices'].cpu().numpy().flatten().tolist()
                    self._code_cache[emb_key] = codes
                    self._using_real_codes = True
                    return codes
            except Exception:
                pass
        
        # Fallback: hash-based pseudo-codes
        self._using_real_codes = False
        codes = []
        for h in range(4):
            section = emb[h*96:(h+1)*96]
            code = int(np.abs(section.sum() * 100) % 256)
            codes.append(code)
        
        self._code_cache[emb_key] = codes
        return codes
    
    def _decode_bridge_node(self, node: Tuple[int, int]) -> np.ndarray:
        """
        Convert graph node to 384D direction vector.
        
        Uses VQ-VAE codebook if available, else sparse vector.
        """
        head, code = node
        
        # 1. Try precomputed vectors (V3 Alignment)
        if self.bridge_vectors is not None:
            try:
                vec = self.bridge_vectors[head][code]
                self._using_real_decode = True
                return vec
            except Exception:
                pass
        
        # 2. Try VQ-VAE Codebook
        if self.vqvae is not None:
            try:
                codebook = self.vqvae.quantizer.codebook[head].weight
                vec = codebook[code].detach().cpu().numpy()
                self._using_real_decode = True
                return vec
            except Exception:
                pass
        
        # Fallback: sparse directed vector
        self._using_real_decode = False
        direction = np.zeros(384)
        base_idx = head * 96
        direction[base_idx + (code % 96)] = 1.0
        direction[(base_idx + code * 3) % 384] = 0.5
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def _apply_bridge_to_current(self, current: np.ndarray, node: Tuple[int, int]) -> np.ndarray:
        """Apply decoded vector to the corresponding slice."""
        head, code = node
        cand = current.copy()
        
        vec = self._decode_bridge_node(node)
        if vec.shape[0] == 96:
            s = head * 96
            cand[s:s+96] = vec
        elif vec.shape[0] == 384:
            cand = vec.copy()
        
        return cand / (np.linalg.norm(cand) + 1e-8)
    
    def _is_generic_hub(self, node: Tuple[int, int]) -> bool:
        """Detect generic hubs (codes 0, 255 that are universal attractors)."""
        _, code = node
        return code in (0, 255)
    
    def _get_specific_neighbors(self, node: Tuple[int, int], top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """Return specific (non-hub) neighbors sorted by weight."""
        if node not in self.mycelial.graph:
            return []
        
        neighbors = []
        for neighbor, weight in self.mycelial.graph[node].items():
            if not self._is_generic_hub(neighbor) and weight > 1.0:
                neighbors.append((neighbor, weight))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    def find_path_between_codes_adaptive(
        self, 
        start_codes: List[int], 
        target_codes: List[int], 
        max_hops: int = 4
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Adaptive BFS: avoids generic hubs, prioritizes specific bridges.
        
        Returns:
            (path, specificity score)
        """
        start_nodes = set((h, int(c)) for h, c in enumerate(start_codes) if h < 4)
        target_nodes = set((h, int(c)) for h, c in enumerate(target_codes) if h < 4)
        
        start_specific = {n for n in start_nodes if not self._is_generic_hub(n)}
        target_specific = {n for n in target_nodes if not self._is_generic_hub(n)}
        
        if not start_specific:
            start_specific = start_nodes
        if not target_specific:
            target_specific = target_nodes
        
        queue = [(node, [node], 0.0) for node in start_specific]
        visited = set(start_specific)
        best_path = []
        best_score = -1.0
        
        while queue:
            queue.sort(key=lambda x: x[2], reverse=True)
            current, path, score = queue.pop(0)
            
            if current in target_specific:
                if score > best_score:
                    best_path = path
                    best_score = score
                continue
            
            if (len(path) - 1) >= max_hops:
                continue
            
            specific_neighbors = self._get_specific_neighbors(current)
            
            for neighbor, weight in specific_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_score = score + np.log1p(weight)
                    queue.append((neighbor, path + [neighbor], new_score))
        
        return best_path, best_score

    def propose(
        self, 
        ctx: Context, 
        start_concept: Optional[str] = None, 
        target_concept: Optional[str] = None
    ) -> NavigationStep:
        """
        Propose next step using Mycelial graph bridges.
        
        ADAPTIVE:
        - Dynamic weight based on initial similarity (more weight when distant)
        - Filters generic hubs (0, 255)
        - Prioritizes specific bridges with high weights
        """
        start_codes = self._get_codes_from_embedding(ctx.start_emb)
        target_codes = self._get_codes_from_embedding(ctx.target_emb)
        
        cache_key = (tuple(start_codes), tuple(target_codes))
        
        if cache_key not in self._bridge_cache:
            bridges, specificity = self.find_path_between_codes_adaptive(start_codes, target_codes)
            self._bridge_cache[cache_key] = (bridges, specificity)
        
        bridges, specificity = self._bridge_cache[cache_key]
        
        current_sim = float(np.dot(ctx.current, ctx.target_emb))
        distance_factor = max(0, 1.0 - current_sim)
        
        if not bridges:
            direction = ctx.target_emb - ctx.current
            confidence = 0.05
            reasoning = "No bridge (fallback)"
        else:
            current_codes = self._get_codes_from_embedding(ctx.current)
            current_nodes = set((h, c) for h, c in enumerate(current_codes))
            
            idx = self._bridge_progress.get(cache_key, 0)
            
            while idx < len(bridges) and (bridges[idx] in current_nodes or self._is_generic_hub(bridges[idx])):
                idx += 1
            
            if idx >= len(bridges):
                idx = len(bridges) - 1
            
            next_bridge = bridges[idx]
            self._bridge_progress[cache_key] = idx
            
            bridge_point = self._apply_bridge_to_current(ctx.current, next_bridge)
            direction = bridge_point - ctx.current
            direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
            
            step_size = 0.15 * (1.0 - 0.4 * (ctx.step / 15.0))
            pred = ctx.current + direction_norm * step_size
            pred = pred / (np.linalg.norm(pred) + 1e-8)
            
            current_sim = float(np.dot(ctx.current, ctx.target_emb))
            pred_sim = float(np.dot(pred, ctx.target_emb))
            delta = pred_sim - current_sim
            
            stagnation_detected = ctx.step > 2 and current_sim < 0.7
            
            threshold = 0.0
            if stagnation_detected:
                threshold = -0.05
            
            if delta <= threshold:
                confidence = 0.05
                reasoning = f"Bridge rejected (delta={delta:+.4f}) h={next_bridge[0]},c={next_bridge[1]}"
            else:
                confidence = min(0.75, 0.35 + 4.0 * delta)
                
                if stagnation_detected and delta > -0.05 and delta <= 0:
                    confidence = max(confidence, 0.45)
                    reasoning = f"Bridge STAGNATION_RESCUE h={next_bridge[0]},c={next_bridge[1]} (Δ={delta:+.4f})"
                elif not (self._using_real_codes and self._using_real_decode):
                    confidence = min(confidence, 0.25)
                    reasoning = f"Bridge (pseudo) h={next_bridge[0]},c={next_bridge[1]} (Δ={delta:+.4f})"
                else:
                    reasoning = f"Bridge precise h={next_bridge[0]},c={next_bridge[1]} (Δ={delta:+.4f})"
        
        return NavigationStep(
            agent_id="mycelial_bridge",
            personality=HeuristicPersonality.MYCELIAL_BRIDGE,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning
        )
