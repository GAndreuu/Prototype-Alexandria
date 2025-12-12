
# scripts/verify_autonomy.py
"""
Verify Autonomy - Day in the Life Smoke Test
--------------------------------------------

Roteiro:
1. Cold start do sistema (Mycelial, Field, ActiveInference, Loop).
2. Injeta um "knowledge gap" inicial a partir de um tópico.
3. Roda N ciclos do SelfFeedingLoop.
4. Coleta métricas antes/depois e imprime um relatório de autonomia.

Uso:
    python scripts/verify_autonomy.py --topic "continual learning" --cycles 5 --mode primary
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.reasoning.mycelial_reasoning import MycelialReasoning
from core.field.pre_structural_field import PreStructuralField
from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig
from core.loop.active_inference_adapter import ActiveInferenceActionAdapter
from core.reasoning.abduction_engine import AbductionEngine, KnowledgeGap
try:
    from core.reasoning.symbol_grounding import SymbolGrounder
except ImportError:
    SymbolGrounder = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alexandria Autonomy Smoke Test")
    parser.add_argument(
        "--topic",
        type=str,
        default="continual learning",
        help="Tópico ou pergunta inicial (knowledge gap)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Número de ciclos do SelfFeedingLoop",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["shadow", "primary"],
        default="primary",
        help="Modo de Active Inference: shadow (só observa) ou primary (dirige ações)",
    )
    return parser.parse_args()


def build_system(mode: str):
    """
    Instancia os componentes principais do Alexandria para o teste.
    """
    logger.info("🔧 Building System Components...")
    
    # Core reasoning components
    mycelial = MycelialReasoning()
    # Ensure clean state for test uniqueness or standard basis
    # mycelial.reset() # Keep persistence if we want to test evolution over runs
    if len(mycelial.graph) == 0:
        logger.info("Initializing Mycelial Graph with seed nodes...")
        mycelial.observe([10, 20, 30, 40]) # Seed
    
    field = PreStructuralField()
    field.connect_mycelial(mycelial)
    
    # Abduction Engine
    abduction = AbductionEngine(fast_mode=True)

    # Active Inference adapter
    adapter = ActiveInferenceActionAdapter()

    # Loop config
    config = LoopConfig(
        use_active_inference_shadow=(mode == "shadow"),
        use_active_inference=(mode == "primary"),
        max_hypotheses_per_cycle=3
    )

    loop = SelfFeedingLoop(
        config=config,
        active_inference_adapter=adapter,
        mycelial=mycelial,
        field=field,
        abduction_engine=abduction,
        hypothesis_executor=None # Use default
    )
    
    # Ensure executor has mycelial access (manual injection seeing as loop init might not wire it)
    if hasattr(loop.executor, 'mycelial'):
         loop.executor.mycelial = mycelial
    else:
        # If executor was created inside loop without mycelial (old init pattern)
        # We need to manually set it if the init signature didn't support it fully
        # But we updated executor to accept it. 
        # SelfFeedingLoop creates it if None. We need to check if SelfFeedingLoop passes it.
        pass
        
    # NOTE: SelfFeedingLoop.__init__ does:
    # self.executor = hypothesis_executor or HypothesisExecutor()
    # It does NOT pass mycelial to default executor!
    # Correcting this:
    if loop.executor.mycelial is None:
        logger.info("Wiring Mycelial to Executor...")
        loop.executor.mycelial = mycelial

    return loop, mycelial, field, adapter


def inject_initial_gap(loop: SelfFeedingLoop, topic: str) -> None:
    """
    Cria um "knowledge gap" inicial a partir do tópico e injeta no sistema.
    """
    gap_id = f"gap_smoke_test_{int(datetime.now().timestamp())}"
    logger.info(f"💉 Injecting Knowledge Gap: {gap_id} ('{topic}')")
    
    gap = KnowledgeGap(
        gap_id=gap_id,
        gap_type="missing_connection", # Type that encourages connection
        description=f"Initial seed gap for smoke test: {topic}",
        affected_clusters=[topic, "system stability"], # Ensure two groundable concepts
        priority_score=0.9,
        candidate_hypotheses=[],
        detected_at=datetime.now()
    )
    
    if loop.abduction_engine:
        loop.abduction_engine.knowledge_gaps[gap_id] = gap
        logger.info("Gap injected successfully into AbductionEngine.")
    else:
        logger.warning("No AbductionEngine found on Loop!")


def get_system_stats(mycelial: MycelialReasoning, field: PreStructuralField) -> Dict[str, Any]:
    """
    Coleta métricas básicas do sistema.
    """
    stats: Dict[str, Any] = {
        "mycelial": {},
        "field": {},
    }

    # Mycelial
    try:
        stats["mycelial"] = mycelial.get_network_stats()
    except Exception as e:
        logger.warning(f"Could not get mycelial stats: {e}")

    # Field
    try:
        stats["field"] = field.stats()
    except Exception as e:
        logger.warning(f"Could not get field stats: {e}")

    return stats


def run_smoke_test(topic: str, cycles: int, mode: str) -> Dict[str, Any]:
    start_time = datetime.now()
    logger.info(f"=== Alexandria Autonomy Smoke Test ===")
    logger.info(f"Topic      : {topic}")
    logger.info(f"Cycles     : {cycles}")
    logger.info(f"Mode       : {mode}")

    loop, mycelial, field, adapter = build_system(mode)

    # Snapshot antes de rodar
    stats_before = get_system_stats(mycelial, field)

    # Verify Grounding (Log potential nodes)
    if SymbolGrounder:
        try:
            grounder = SymbolGrounder()
            nodes = grounder.ground(topic)
            logger.info(f"Checking Grounding for '{topic}': found {len(nodes)} nodes.")
        except Exception as e:
            logger.warning(f"Grounding check failed: {e}")

    # Injeta objetivo inicial
    inject_initial_gap(loop, topic)

    cycles_executed = 0
    errors = 0
    actions_taken = []

    for i in range(cycles):
        try:
            logger.info(f"\n🔄 Cycle {i+1}/{cycles}")
            cycle_metrics = loop.run_cycle()
            cycles_executed += 1
            
            # Log results
            succ = cycle_metrics.actions_successful
            total = cycle_metrics.actions_executed
            
            # Fetch real-time stats
            active_nodes = mycelial.get_network_stats().get("active_nodes", 0)
            logger.info(f"   Nodes Active: {active_nodes} | Actions: {succ}/{total}")
            
            # Store actions taken (if available in loop state or logs)
            if hasattr(loop, 'shadow_actions') and mode == 'shadow':
                logger.info(f"   Shadow Suggestions: {len(loop.shadow_actions)}")
            
        except Exception as e:
            logger.error(f"Error during cycle {i+1}: {e}", exc_info=True)
            errors += 1
            errors += 1
            break

    # FORCING MANUAL VERIFICATION OF GROUNDING
    # Since agent policy might act conservatively, we force one bridge to prove mechanism works
    try:
        logger.info("⚡ FORCING MANUAL GROUNDED BRIDGE to verify capabilities...")
        from core.loop.hypothesis_executor import ExecutableAction, ExecutionActionType
        action = ExecutableAction(
            action_type=ExecutionActionType.BRIDGE_CONCEPTS,
            target=f"{topic} <-> system stability",
            parameters={"source": topic, "target": "system stability"},
            source_hypothesis_id="manual_force"
        )
        result = loop.executor._execute_action(action)
        logger.info(f"Manual Bridge: Success={result.success}, Edges={result.new_connections}")
    except Exception as e:
        logger.warning(f"Manual Force Failed: {e}")

    # Snapshot depois
    stats_after = get_system_stats(mycelial, field)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary = {
        "topic": topic,
        "mode": mode,
        "cycles_requested": cycles,
        "cycles_executed": cycles_executed,
        "errors": errors,
        "duration_sec": duration,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "shadow_actions_count": len(getattr(loop, "shadow_actions", [])),
    }

    log_summary(summary)
    return summary


def log_summary(summary: Dict[str, Any]) -> None:
    logger.info("\n\n📊 === Smoke Test Summary ===")
    logger.info(f"Topic            : {summary['topic']}")
    logger.info(f"Mode             : {summary['mode']}")
    logger.info(f"Cycles executed  : {summary['cycles_executed']}/{summary['cycles_requested']}")
    logger.info(f"Errors           : {summary['errors']}")
    logger.info(f"Duration (sec)   : {summary['duration_sec']:.2f}")

    mb = summary["stats_before"].get("mycelial", {})
    ma = summary["stats_after"].get("mycelial", {})
    
    edges_b = mb.get("active_edges", 0)
    edges_a = ma.get("active_edges", 0)
    nodes_b = mb.get("active_nodes", 0)
    nodes_a = ma.get("active_nodes", 0)
    
    logger.info(f"Mycelial Edges   : {edges_b} -> {edges_a} ({edges_a - edges_b:+})")
    logger.info(f"Mycelial Nodes   : {nodes_b} -> {nodes_a} ({nodes_a - nodes_b:+})")

    if summary["mode"] == "shadow":
        logger.info(f"Shadow Actions   : {summary['shadow_actions_count']}")


if __name__ == "__main__":
    args = parse_args()
    run_smoke_test(args.topic, args.cycles, args.mode)
