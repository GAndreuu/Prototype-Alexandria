"""
Demo: Self-Feeding Loop
========================

Script de demonstração do ciclo auto-alimentado.
Roda em modo simulado (sem abduction_engine real).

Uso:
    python scripts/demos/demo_self_feeding_loop.py
"""

import sys
import os

# Adicionar raiz do projeto ao path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from core.loop.hypothesis_executor import HypothesisExecutor
from core.loop.feedback_collector import ActionFeedbackCollector
from core.loop.incremental_learner import IncrementalLearner
from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig

def main():
    print("=" * 60)
    print("🔄 SELF-FEEDING LOOP - DEMO")
    print("=" * 60)
    
    # Configuração
    config = LoopConfig(
        max_cycles=10,
        max_hypotheses_per_cycle=3,
        stop_on_convergence=False,  # Rodar todos os ciclos
        log_every_n_cycles=1,
        metrics_save_path="data/demo_loop_metrics.json"
    )
    
    # Criar componentes
    executor = HypothesisExecutor()
    collector = ActionFeedbackCollector()
    learner = IncrementalLearner(batch_threshold=5)
    
    # Criar loop
    loop = SelfFeedingLoop(
        hypothesis_executor=executor,
        feedback_collector=collector,
        incremental_learner=learner,
        config=config
    )
    
    print("\n📊 Configuração:")
    print(f"   Max cycles: {config.max_cycles}")
    print(f"   Hypotheses/cycle: {config.max_hypotheses_per_cycle}")
    print(f"   Batch threshold: {learner.batch_threshold}")
    
    print("\n🚀 Iniciando loop...\n")
    
    # Executar
    results = loop.run_continuous()
    
    # Resultados
    print("\n" + "=" * 60)
    print("📈 RESULTADOS")
    print("=" * 60)
    
    print(f"\n✅ Ciclos executados: {results['cycles_run']}")
    print(f"⏱️  Tempo total: {results['total_time_seconds']:.2f}s")
    print(f"🎯 Convergiu: {results['converged']}")
    
    summary = results['metrics_summary']
    print(f"\n📊 Métricas:")
    print(f"   Total de ações: {summary['total_actions']}")
    print(f"   Taxa de sucesso: {summary['success_rate']:.1%}")
    print(f"   Evidências encontradas: {summary['total_evidence']}")
    print(f"   Conexões criadas: {summary['total_connections']}")
    print(f"   Eventos de aprendizado: {summary['total_learning_events']}")
    print(f"   Reward acumulado: {summary['cumulative_reward']:.2f}")
    print(f"   Score de convergência: {summary['convergence_score']:.2f}")
    
    executor_stats = results['executor_stats']
    print(f"\n🔧 Executor:")
    print(f"   Execuções: {executor_stats['total_executions']}")
    print(f"   Sucesso: {executor_stats['success_rate']:.1%}")
    
    learner_stats = results['learner_stats']
    print(f"\n🧠 Learner:")
    print(f"   Total aprendido: {learner_stats['total_learned']} embeddings")
    print(f"   Última loss: {learner_stats['last_loss']:.4f}")
    
    print(f"\n💾 Métricas salvas em: {config.metrics_save_path}")
    print("\n" + "=" * 60)
    print("✅ DEMO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()
