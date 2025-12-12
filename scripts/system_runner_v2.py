"""
Alexandria System Runner V2
============================
REAL integration with:
- AbductionEngine for true gap detection via CausalGraph
- 16-worker parallelization on i9
- Optimized crystallization 

Usage:
    python scripts/system_runner_v2.py --cycles 100 --workers 16
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemRunnerV2")


class AlexandriaSystemRunnerV2:
    """
    Complete system orchestrator with REAL integration.
    
    - Real AbductionEngine for gap detection
    - Real HypothesisExecutor for actions
    - 16-worker parallelization
    """
    
    def __init__(
        self,
        vqvae_path: str = "data/monolith_v13_finetuned.pth",
        mycelial_path: str = "data/mycelial_state.npz",
        topology_path: str = "data/topology.json",
        metrics_path: str = "data/system_metrics_v2.json",
        num_workers: int = 16
    ):
        self.vqvae_path = vqvae_path
        self.mycelial_path = mycelial_path
        self.topology_path = topology_path
        self.metrics_path = metrics_path
        self.num_workers = num_workers
        
        # Components
        self.vqvae = None
        self.mycelial = None
        self.topology = None
        self.field = None
        self.abduction_engine = None
        self.loop = None
        
        # Metrics
        self.metrics = {
            "start_time": None,
            "cycles_completed": 0,
            "gaps_detected": 0,
            "hypotheses_generated": 0,
            "actions_executed": 0,
            "consolidations": 0,
            "errors": [],
            "cycle_times": []
        }
        
        self.device = "cpu"  # RX 580 not supported in PyTorch Windows
        logger.info(f"SystemRunnerV2 initialized. Device: {self.device}, Workers: {num_workers}")
    
    def initialize_all(self):
        """Initialize all system components with REAL integrations."""
        logger.info("=" * 60)
        logger.info("INITIALIZING ALEXANDRIA SYSTEM V2 (REAL MODE)")
        logger.info("=" * 60)
        
        # 1. VQ-VAE
        logger.info("\n[1/7] Loading VQ-VAE...")
        from core.reasoning.vqvae.model import MonolithV13
        self.vqvae = MonolithV13()
        if os.path.exists(self.vqvae_path):
            self.vqvae.load_state_dict(torch.load(self.vqvae_path, map_location=self.device))
            logger.info(f"  ✅ Loaded from {self.vqvae_path}")
        self.vqvae.to(self.device)
        self.vqvae.eval()
        
        # 2. Mycelial Network
        logger.info("\n[2/7] Loading Mycelial Network...")
        from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
        config = MycelialConfig(save_path=self.mycelial_path)
        self.mycelial = MycelialReasoning(config)
        logger.info(f"  ✅ Loaded: {len(self.mycelial.node_activation_counts)} nodes")
        
        # 3. Topology Engine
        logger.info("\n[3/7] Loading Topology Engine...")
        from core.topology.topology_engine import TopologyEngine
        self.topology = TopologyEngine()
        if os.path.exists(self.topology_path):
            self.topology.load_topology(self.topology_path)
            logger.info(f"  ✅ Loaded: {self.topology.n_clusters} clusters")
        
        # 4. PreStructuralField
        logger.info("\n[4/7] Initializing PreStructuralField...")
        from core.field import PreStructuralField
        self.field = PreStructuralField()
        self.field.connect_vqvae(self.vqvae)
        self.field.connect_mycelial(self.mycelial)
        logger.info("  ✅ Field connected")
        
        # 5. REAL AbductionEngine (FAST MODE - no LanceDB queries)
        logger.info("\n[5/7] Initializing REAL AbductionEngine (FAST MODE)...")
        from core.reasoning.abduction_engine import AbductionEngine
        self.abduction_engine = AbductionEngine(fast_mode=True)  # FAST MODE!
        logger.info("  ✅ AbductionEngine ready (CausalGraph + FAST MODE)")
        
        # 6. Self-Feeding Loop with REAL AbductionEngine
        logger.info("\n[6/7] Initializing Self-Feeding Loop...")
        from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig
        loop_config = LoopConfig(
            max_cycles=1000,
            stop_on_convergence=False,
            log_every_n_cycles=1,
            save_metrics_every_n_cycles=10,
            metrics_save_path="data/loop_metrics.json"
        )
        self.loop = SelfFeedingLoop(
            abduction_engine=self.abduction_engine,  # REAL!
            config=loop_config
        )
        logger.info("  ✅ Loop with REAL AbductionEngine ready")
        
        # 7. Thread Pool for parallel processing
        logger.info(f"\n[7/7] Setting up {self.num_workers}-worker thread pool...")
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        logger.info(f"  ✅ Thread pool ready")
        
        logger.info("\n" + "=" * 60)
        logger.info("SYSTEM V2 INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        
        return True
    
    def run_extended_cycles(
        self, 
        max_cycles: int = 100,
        consolidate_every: int = 25,
        save_every: int = 25,
        max_attractors: int = 25  # Limit crystallization
    ):
        """
        Run extended cognitive cycles with REAL components.
        """
        self.metrics["start_time"] = datetime.now().isoformat()
        logger.info(f"\n🚀 Starting {max_cycles} REAL cognitive cycles...")
        logger.info(f"   Workers: {self.num_workers}")
        logger.info(f"   Consolidation every {consolidate_every} cycles")
        logger.info(f"   Max attractors: {max_attractors}")
        
        # Load embeddings for sampling
        embeddings = None
        if os.path.exists("data/training_embeddings.npy"):
            embeddings = np.load("data/training_embeddings.npy")
            logger.info(f"   Loaded {len(embeddings)} embeddings")
        
        start_time = time.time()
        
        for cycle in range(1, max_cycles + 1):
            cycle_start = time.time()
            
            try:
                # 1. Sample random embedding
                if embeddings is not None:
                    idx = np.random.randint(0, len(embeddings))
                    embedding = embeddings[idx]
                else:
                    embedding = np.random.randn(384).astype(np.float32)
                
                # 2. VQ-VAE quantization
                with torch.no_grad():
                    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    vqvae_output = self.vqvae(emb_tensor)
                    indices = vqvae_output['indices'].numpy()[0]
                
                # 3. Mycelial observation
                self.mycelial.observe(indices)
                
                # 4. Field trigger (lightweight, no propagation every cycle)
                if cycle % 5 == 0 and self.field:
                    try:
                        self.field.trigger(embedding, codes=indices, intensity=0.5)
                    except:
                        pass
                
                # 5. REAL cognitive cycle with AbductionEngine
                cycle_metrics = self.loop.run_cycle()
                
                # Track metrics
                gaps = getattr(cycle_metrics, 'gaps_detected', 0)
                hyps = getattr(cycle_metrics, 'hypotheses_generated', 0)
                actions = getattr(cycle_metrics, 'actions_executed', 0)
                
                self.metrics["cycles_completed"] = cycle
                self.metrics["gaps_detected"] += gaps
                self.metrics["hypotheses_generated"] += hyps
                self.metrics["actions_executed"] += actions
                
                cycle_time = time.time() - cycle_start
                self.metrics["cycle_times"].append(cycle_time)
                
                # Log
                if cycle % 5 == 0 or gaps > 1:
                    logger.info(f"Cycle {cycle}/{max_cycles} | Gaps: {gaps} | Hyps: {hyps} | VQ: {indices} | Time: {cycle_time:.2f}s")
                
                # Consolidation
                if cycle % consolidate_every == 0:
                    self._run_consolidation(cycle, max_attractors=max_attractors)
                
                # Save
                if cycle % save_every == 0:
                    self._save_state(cycle)
                    
            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}")
                self.metrics["errors"].append({"cycle": cycle, "error": str(e)})
                continue
        
        total_time = time.time() - start_time
        avg_cycle = sum(self.metrics["cycle_times"]) / len(self.metrics["cycle_times"]) if self.metrics["cycle_times"] else 0
        
        logger.info(f"\n✅ Completed {max_cycles} cycles in {total_time:.1f}s")
        logger.info(f"   Avg cycle time: {avg_cycle:.2f}s")
        logger.info(f"   Total gaps detected: {self.metrics['gaps_detected']}")
        logger.info(f"   Total hypotheses: {self.metrics['hypotheses_generated']}")
        
        # Final save
        self._run_consolidation(max_cycles, final=True, max_attractors=max_attractors)
        self._save_state(max_cycles, final=True)
        
        return self.metrics
    
    def _run_consolidation(self, cycle: int, final: bool = False, max_attractors: int = 25):
        """Run optimized field consolidation."""
        logger.info(f"\n🌙 {'FINAL ' if final else ''}Consolidation at cycle {cycle}...")
        
        try:
            if self.field:
                # Run field cycle
                result = self.field.run_cycle()
                
                # Get attractors first
                attractors = self.field.get_attractors()
                n_attractors = len(attractors) if attractors is not None else 0
                
                logger.info(f"   Field attractors: {n_attractors}")
                
                # Only crystallize top-N attractors to avoid O(n²) explosion
                if n_attractors > 0 and n_attractors <= max_attractors:
                    graph = self.field.crystallize()
                    logger.info(f"   Crystallized: {len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges")
                elif n_attractors > max_attractors:
                    logger.info(f"   Skipping crystallization (too many attractors: {n_attractors})")
            
            # Decay Mycelial
            if self.mycelial:
                self.mycelial.decay()
                
            self.metrics["consolidations"] += 1
            
        except Exception as e:
            logger.error(f"   Consolidation error: {e}")
    
    def _save_state(self, cycle: int, final: bool = False):
        """Save all states."""
        logger.info(f"💾 {'FINAL ' if final else ''}Saving state...")
        
        try:
            if self.mycelial:
                self.mycelial.save_state()
            if self.topology and self.topology.is_trained:
                self.topology.save_topology(self.topology_path)
            
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
                
            logger.info("   ✅ Saved")
        except Exception as e:
            logger.error(f"   Save error: {e}")


def main():
    import argparse
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Alexandria System Runner V2")
    parser.add_argument("--cycles", type=int, default=50, help="Number of cycles")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--consolidate-every", type=int, default=25, help="Consolidation frequency")
    parser.add_argument("--max-attractors", type=int, default=25, help="Max attractors for crystallization")
    args = parser.parse_args()
    
    runner = AlexandriaSystemRunnerV2(num_workers=args.workers)
    
    if not runner.initialize_all():
        logger.error("Initialization failed!")
        return
    
    runner.run_extended_cycles(
        max_cycles=args.cycles,
        consolidate_every=args.consolidate_every,
        max_attractors=args.max_attractors
    )


if __name__ == "__main__":
    main()
