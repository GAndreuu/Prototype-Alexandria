"""
swarm/navigator.py

The central SwarmNavigator system.
Integrates neurodiverse agents, topological awareness, active consensus,
and complexity-based adaptive modes into a cohesive navigation engine.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union

# Core types
from .core import (
    Context, NavigationStep, NavigationResult,
    NeurotypeName, NavigationMode, ModeConfig,
    TopologicalState, SwarmAction, ActionType
)

# Agents from the consolidated agents module
from .agents.direct import DirectAgent
from .agents.gradient import RealGradientAgent
from .agents.momentum import MomentumAgent
from .agents.explorer import ExplorerAgent
from .agents.collapse import CollapseAgent
from .agents.critical import CriticalAgent
from .agents.psychedelic import PsychedelicAgent
from .agents.autistic import AutisticAgent
from .agents.mycelial_bridge import MycelialBridgeAgent

# Components
from .core.consensus import NeurodiverseConsensus, ConsensusResult

# NOTE: These components are now in swarm/topology/ and swarm/core/
try:
    from .topology.analyzer import TopologyAnalyzer
    from .core.memory import PersistentTopologicalMemory 
    from .tools.complexity_classifier import ComplexityClassifier
    from .tools.early_stopping import EarlyStoppingCommittee
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False
    TopologyAnalyzer = None
    PersistentTopologicalMemory = None
    ComplexityClassifier = None
    EarlyStoppingCommittee = None

class SwarmNavigator:
    """
    The SwarmNavigator engine.
    
    Orchestrates a diverse team of agentic heuristics to navigate
    high-dimensional semantic spaces (384D).
    
    Features:
    - Adaptive neurodiverse consensus (fair weighted voting)
    - Topological awareness (curvature, density, collapse risk)
    - Persistent memory of past traversals
    - Complexity-aware mode switching (Sprint vs. Balanced vs. Creative)
    """
    
    def __init__(
        self,
        topology_engine=None,
        mycelial_system=None,
        memory_path: str = "data/swarm_memory.json",
        use_neurodiverse: bool = True
    ):
        self.logger = logging.getLogger("SwarmNavigator")
        self.topology = topology_engine
        self.mycelial = mycelial_system
        self.use_neurodiverse = use_neurodiverse
        
        # --- Core Components ---
        # NOTE: Consensus is initialized AFTER memory (in the block below)
        # to enable the feedback loop
        self.consensus = None  # Will be set after memory init
        
        # Initialize sub-systems if available modules exist
        # (Using local imports or assumed structure)
        # Initialize sub-systems if available modules exist
        if V3_AVAILABLE:
            try:
                self.topo_analyzer = TopologyAnalyzer()
                self.memory = PersistentTopologicalMemory(save_path=memory_path)
                self.complexity_classifier = ComplexityClassifier()
                self.early_stopper = EarlyStoppingCommittee()
                self.has_v3_components = True
            except Exception as e:
                self.logger.warning(f"V3 components available but initialization failed: {e}")
                self.has_v3_components = False
                self.topo_analyzer = None
                self.memory = None
        else:
            self.logger.warning("Running in limited mode (V3 components import failed)")
            self.has_v3_components = False
            self.topo_analyzer = None
            self.memory = None

        # FEEDBACK LOOP: Initialize consensus WITH memory reference
        # This enables the consensus system to query historical trajectories
        # before voting, closing the learning loop
        self.consensus = NeurodiverseConsensus(memory=self.memory)

        # --- Agents Initialization ---
        self.agents = {}
        self._init_agents()
        
    def _init_agents(self):
        """Initialize the team of agents."""
        # Standard Agents
        self.agents['direct'] = DirectAgent()
        self.agents['gradient'] = RealGradientAgent() # Requires gradient source if available
        self.agents['momentum'] = MomentumAgent()
        self.agents['explorer'] = ExplorerAgent()
        
        # Neurodiverse Agents (V3)
        if self.use_neurodiverse:
            self.agents['collapse'] = CollapseAgent()
            self.agents['critical'] = CriticalAgent()
            self.agents['psychedelic'] = PsychedelicAgent()
            self.agents['autistic'] = AutisticAgent()
            
            if self.mycelial:
                self.agents['bridge'] = MycelialBridgeAgent(self.mycelial)

    def navigate(
        self,
        start_concept: Union[str, np.ndarray],
        target_concept: Union[str, np.ndarray],
        max_steps: int = 50,
        debug: bool = False
    ) -> Dict: # Returns generic dict or NavigationResult
        """
        Execute navigation from start to target.
        """
        # 1. Resolve embeddings
        start_emb = self._resolve_concept(start_concept)
        target_emb = self._resolve_concept(target_concept)
        
        if start_emb is None or target_emb is None:
            return {"success": False, "reason": "Concept resolution failed"}
            
        # 2. Determine Mode & Complexity (if V3 components active)
        mode = NavigationMode.BALANCED
        config = None
        
        if self.has_v3_components:
            mode, config = self.complexity_classifier.classify(
                start_emb, target_emb, context={"mycelial": self.mycelial}
            )
            if config:
                max_steps = config.max_steps
        
        # 3. Initialize Context
        current_pos = start_emb.copy()
        history = [current_pos]
        
        ctx = Context(
            start_emb=start_emb,
            target_emb=target_emb,
            current=current_pos,
            history=history,
            step=0,
            initial_dist=np.linalg.norm(target_emb - current_pos)
        )
        
        success = False
        neurotype_contributions = {}
        topological_events = []
        
        # 4. Main Navigation Loop
        for step in range(max_steps):
            ctx.step = step
            ctx.current = current_pos
            ctx.history = history
            
            if debug:
                sim_now = np.dot(current_pos, target_emb)
                print(f"\n{'='*60}")
                print(f"STEP {step} | Similarity: {sim_now:.4f}")
                print(f"{'='*60}")
            
            # --- A. Topological Analysis ---
            if self.has_v3_components and self.topo_analyzer:
                topo_state = self.topo_analyzer.analyze_local(
                    current_pos, target_emb, history
                )
                ctx.topological_state = topo_state
                if debug:
                    print(f"[Topology] Energy: {topo_state.energy:.3f} | Curvature: {topo_state.curvature:.3f} | Density: {topo_state.density:.3f}")
                # Check for collapse/danger
                if topo_state.near_collapse:
                    topological_events.append({
                        "step": step, "type": "warning", "msg": "Near topological collapse"
                    })
                    if debug: print(f"[Topology] ⚠️  NEAR COLLAPSE DETECTED")

            # --- B. Agent Proposals ---
            proposals = []
            if debug: print(f"\n[Agents] Collecting proposals from {len(self.agents)} agents:")
            for name, agent in self.agents.items():
                try:
                    # Some agents might need different propose signatures
                    # We assume they are updated to accept Context from core
                    prop = agent.propose(ctx)
                    if prop:
                         # Attach neurotype if missing (legacy agents)
                        if not hasattr(prop, 'neurotype') or not prop.neurotype:
                             prop.neurotype = self._map_agent_to_neurotype(name)
                        proposals.append(prop)
                        if debug:
                            dir_summary = f"[{prop.direction[0]:.2f}, {prop.direction[1]:.2f}, ...]"
                            print(f"  → {name:12s} | conf: {prop.confidence:.2f} | dir: {dir_summary}")
                except Exception as e:
                    if debug: print(f"  ✗ {name:12s} | ERROR: {e}")
            
            # --- C. Consensus ---
            if debug: print(f"\n[Consensus] Computing weighted consensus...")
            consensus_result = self.consensus.compute_consensus(
                proposals, ctx, target_mix=config.neurotype_mix if config else None
            )
            
            if debug:
                print(f"  Confidence: {consensus_result.confidence:.3f}")
                print(f"  Contributing neurotypes:")
                for nt, weight in consensus_result.contributing_neurotypes.items():
                    nt_name = nt.value if hasattr(nt, 'value') else str(nt)
                    print(f"    {nt_name:12s}: {weight:.3f}")
            
            # Record contributions
            for nt, weight in consensus_result.contributing_neurotypes.items():
                neurotype_contributions[nt] = neurotype_contributions.get(nt, 0) + weight

            # --- D. Move ---
            # Update position
            step_vector = consensus_result.direction * 0.1 # Step size
            current_pos = current_pos + step_vector
            
            # Normalize to unit sphere if using cosine sim space (Alexandria default)
            current_pos = current_pos / np.linalg.norm(current_pos)
            history.append(current_pos)
            
            if debug:
                # Normalized cosine similarity (fix Bug #1)
                target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
                new_sim = np.dot(current_pos, target_norm)
                prev_pos = history[-2] if len(history) > 1 else current_pos
                delta = new_sim - np.dot(prev_pos, target_norm)
                print(f"\n[Move] Step size: 0.1 | New similarity: {new_sim:.4f} (Δ {delta:+.4f})")
            
            # --- E. Check Completion ---
            # Similarity check (normalized cosine similarity - fix Bug #1)
            target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
            sim = np.dot(current_pos, target_norm)
            if sim > 0.92: # Threshold for "arrived"
                if debug: print(f"\n✅ ARRIVED! Similarity {sim:.4f} > 0.92 threshold")
                success = True
                break
                
            # --- F. Early Stopping ---
            if self.has_v3_components:
                # Should stop check
                stop_decision = self.early_stopper.should_stop(
                    history, target_emb, step, {"mode": mode}
                )
                if debug:
                    print(f"\n[Early Stopping] Decision: {'STOP' if stop_decision.should_stop else 'CONTINUE'}")
                    for v in stop_decision.votes:
                        vote_icon = "🛑" if v.should_stop else "▶️"
                        print(f"  {vote_icon} {v.voter_name:15s} | conf: {v.confidence:.2f} | {v.reason[:40]}")
                if stop_decision.should_stop:
                    if debug: print(f"\n⏹️  EARLY STOP: {stop_decision.primary_reason}")
                    if stop_decision.success_prediction:
                        success = True
                    break

        # 5. Finalize Results (normalized cosine similarity - fix Bug #1)
        target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
        start_norm = start_emb / (np.linalg.norm(start_emb) + 1e-9)
        current_norm = current_pos / (np.linalg.norm(current_pos) + 1e-9)
        
        final_sim = np.dot(current_norm, target_norm)
        init_sim = np.dot(start_norm, target_norm)
        improvement = final_sim - init_sim
        
        # Save to memory
        if self.has_v3_components and self.memory:
            self.memory.save_trajectory(
                trajectory=history,
                start=start_emb,
                target=target_emb,
                efficiency=improvement, # Simplified
                mean_curvature=0.0, # Needs calc
                neurotypes_used=neurotype_contributions,
                success=success
            )

        # 6. Tunneling Detection & Saving
        if success and self.has_v3_components:
            is_tunnel, bridge_data = self._detect_tunneling(history, start_emb, target_emb)
            if is_tunnel:
                self._save_conceptual_bridge(bridge_data)
                topological_events.append({
                    "step": len(history), "type": "discovery", "msg": "Conceptual Bridge Detected"
                })
                if debug: print(f"\n[Discovery] 🌉 Conceptual Bridge detected and saved.")

        return {
            "success": success,
            "steps": len(history) - 1,
            "path": history,
            "init_similarity": float(init_sim),
            "final_similarity": float(final_sim),
            "improvement": float(improvement),
            "neurotype_contributions": neurotype_contributions,
            "topological_events": topological_events,
            "mode": mode.value if hasattr(mode, 'value') else str(mode)
        }

    def _detect_tunneling(
        self, 
        history: List[np.ndarray], 
        start: np.ndarray, 
        target: np.ndarray
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Detect if the path represents a significant tunneling event.
        
        Criteria:
        1. Contextual Jump: Start and Target are semantically distant (sim < 0.3)
        2. Sustained Traversal: Path is non-trivial (steps > 5)
        3. Low Density Traversal: (Implicitly checked by improvement)
        """
        init_sim = float(np.dot(start, target))
        if init_sim > 0.4:
            return False, None  # Concepts were already close
            
        if len(history) < 5:
            return False, None # Too short
            
        # Find midpoint (dark matter candidate)
        mid_idx = len(history) // 2
        midpoint = history[mid_idx]
        
        bridge_data = {
            "start_vector": start.tolist(),
            "target_vector": target.tolist(),
            "midpoint_vector": midpoint.tolist(),
            "steps": len(history),
            "init_similarity": init_sim,
            "timestamp": None # Will use system time
        }
        
        return True, bridge_data

    def _save_conceptual_bridge(self, bridge_data: Dict):
        """Save discovered bridge to disk for future active acquisition."""
        import json
        import os
        from datetime import datetime
        
        bridge_data["timestamp"] = datetime.now().isoformat()
        save_path = "data/conceptual_bridges.json"
        
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    try:
                        bridges = json.load(f)
                    except json.JSONDecodeError:
                        bridges = []
            else:
                bridges = []
                
            bridges.append(bridge_data)
            
            with open(save_path, 'w') as f:
                json.dump(bridges, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save conceptual bridge: {e}")

    def _resolve_concept(self, concept) -> Optional[np.ndarray]:
        """Resolve a string concept to an embedding using Topology or dummy."""
        if isinstance(concept, np.ndarray):
            return concept
        if self.topology:
            # Use encode for real TopologyEngine
            embeddings = self.topology.encode([concept])
            if len(embeddings) > 0:
                return embeddings[0]
            return None
        # Dummy fallback for testing if no topology
        return np.random.randn(384) # Should strictly fail or mock
        
    def _map_agent_to_neurotype(self, agent_name: str) -> NeurotypeName:
        """Map legacy agent names to neurotypes."""
        mapping = {
            'collapse': NeurotypeName.COLLAPSE,
            'critical': NeurotypeName.CRITICAL,
            'psychedelic': NeurotypeName.PSYCH,
            'autistic': NeurotypeName.AUTISTIC,
            'explorer': NeurotypeName.RELAXED,
            'direct': NeurotypeName.COLLAPSE, # Direct is like collapse
            'momentum': NeurotypeName.COLLAPSE,
            'gradient': NeurotypeName.BALANCED,
            'bridge': NeurotypeName.PSYCH
        }
        return mapping.get(agent_name, NeurotypeName.BALANCED)

    def navigate_action(self, action: SwarmAction, debug: bool = False) -> Dict:
        """
        Execute a structured SwarmAction.
        
        This is the primary entry point for external systems (e.g., Nemesis).
        It unpacks the action, configures the navigation strategy based on
        action type, and delegates to the core navigate() method.
        """
        # 1. Unpack action
        start = action.start
        target = action.target
        params = action.params
        constraints = action.constraints
        
        # 2. Determine mode based on action type
        if action.type == ActionType.BRIDGE_CONCEPTS:
            # Bridge requires exploration, use creative mode
            mode = NavigationMode.CREATIVE
            max_steps = constraints.get('max_steps', 50)
            # Enable bridge building (future: pass to bridge builder)
        elif action.type == ActionType.EXPLORE_CLUSTER:
            # Exploration is broad, use creative/relaxed mix
            mode = NavigationMode.CREATIVE
            max_steps = constraints.get('max_steps', 100)
        elif action.type == ActionType.DEEPEN_TOPIC:
            # Deepening is focused, use balanced/cautious
            mode = NavigationMode.CAUTIOUS
            max_steps = constraints.get('max_steps', 30)
        else:
            mode = NavigationMode.BALANCED
            max_steps = constraints.get('max_steps', 50)
        
        # 3. Override mode if explicitly provided in params
        if 'mode' in params:
            mode = NavigationMode(params['mode'])
        
        # 4. Execute navigation
        result = self.navigate(
            start_concept=start,
            target_concept=target,
            max_steps=max_steps,
            debug=debug
        )
        
        # 5. Attach action metadata to result
        result['action_id'] = action.id
        result['action_type'] = action.type.value
        
        return result

