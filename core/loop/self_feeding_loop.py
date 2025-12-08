"""
Self-Feeding Loop - Orquestrador do ciclo auto-alimentado
==========================================================

Conecta todos os componentes:
abduction → action → feedback → learning → (loop)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .hypothesis_executor import HypothesisExecutor, ActionResult
from .feedback_collector import ActionFeedbackCollector
from .incremental_learner import IncrementalLearner
from .loop_metrics import LoopMetrics, CycleMetrics

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuração do Self-Feeding Loop"""
    max_hypotheses_per_cycle: int = 5
    max_cycles: int = 100
    stop_on_convergence: bool = True
    convergence_threshold: float = 0.01
    min_confidence_threshold: float = 0.1
    log_every_n_cycles: int = 1
    save_metrics_every_n_cycles: int = 10
    metrics_save_path: str = "data/loop_metrics.json"


class SelfFeedingLoop:
    """
    Orquestrador do Self-Feeding Loop.
    
    Ciclo:
    1. Detecta knowledge gaps (abduction_engine)
    2. Gera hipóteses para cada gap
    3. Executa ações via HypothesisExecutor
    4. Coleta feedback via FeedbackCollector
    5. Dispara aprendizado via IncrementalLearner
    6. Registra métricas
    7. Repete até convergência ou max_cycles
    """
    
    def __init__(
        self,
        abduction_engine=None,
        hypothesis_executor: Optional[HypothesisExecutor] = None,
        feedback_collector: Optional[ActionFeedbackCollector] = None,
        incremental_learner: Optional[IncrementalLearner] = None,
        config: Optional[LoopConfig] = None,
        on_cycle_complete: Optional[Callable] = None,
        on_action_complete: Optional[Callable] = None
    ):
        self.abduction_engine = abduction_engine
        self.executor = hypothesis_executor or HypothesisExecutor()
        self.collector = feedback_collector or ActionFeedbackCollector()
        self.learner = incremental_learner or IncrementalLearner()
        self.config = config or LoopConfig()
        self.metrics = LoopMetrics()
        self.on_cycle_complete = on_cycle_complete
        self.on_action_complete = on_action_complete
        
        # Estado
        self.is_running = False
        self.current_cycle = 0
        self.last_run_summary: Dict[str, Any] = {}
        
        logger.info("SelfFeedingLoop inicializado")
    
    def run_cycle(self) -> CycleMetrics:
        """
        Executa um único ciclo do loop.
        
        Returns:
            CycleMetrics com dados do ciclo
        """
        start_time = time.time()
        cycle = self.metrics.start_cycle()
        
        try:
            # 1. Detectar gaps
            gaps = self._detect_gaps()
            cycle.gaps_detected = len(gaps)
            
            # 2. Gerar hipóteses
            hypotheses = self._generate_hypotheses(gaps)
            cycle.hypotheses_generated = len(hypotheses)
            
            # 3. Filtrar por confiança mínima
            hypotheses = [
                h for h in hypotheses 
                if h.get("confidence_score", 0) >= self.config.min_confidence_threshold
            ][:self.config.max_hypotheses_per_cycle]
            
            # 4. Executar ações
            total_evidence = 0
            total_connections = 0
            rewards = []
            
            for hypothesis in hypotheses:
                result = self.executor.execute(hypothesis)
                cycle.actions_executed += 1
                
                if result.success:
                    cycle.actions_successful += 1
                
                total_evidence += len(result.evidence_found)
                total_connections += result.new_connections
                
                # 5. Coletar feedback
                feedback = self.collector.collect(result.to_dict())
                rewards.append(feedback.reward_signal)

                # Callback de ação
                if self.on_action_complete:
                    self.on_action_complete(hypothesis, result, feedback)
                
                # 6. Adicionar ao learner
                learned = self.learner.add_feedback(feedback.to_dict())
                if learned:
                    cycle.learning_triggered = True
                    cycle.loss = self.learner.last_loss
            
            cycle.total_evidence = total_evidence
            cycle.new_connections = total_connections
            cycle.avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
        except Exception as e:
            logger.error(f"Erro no ciclo {cycle.cycle_id}: {e}")
        
        cycle.cycle_time_ms = (time.time() - start_time) * 1000
        
        # Registrar métricas
        self.metrics.record_cycle(cycle)
        self.current_cycle += 1
        
        # Callback
        if self.on_cycle_complete:
            self.on_cycle_complete(cycle)
        
        return cycle
    
    def run_continuous(
        self,
        max_cycles: Optional[int] = None,
        stop_on_convergence: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executa múltiplos ciclos até convergência ou limite.
        
        Args:
            max_cycles: Máximo de ciclos (default: config.max_cycles)
            stop_on_convergence: Parar se convergir (default: config.stop_on_convergence)
        
        Returns:
            Dict com resumo da execução
        """
        max_cycles = max_cycles or self.config.max_cycles
        stop_on_convergence = stop_on_convergence if stop_on_convergence is not None else self.config.stop_on_convergence
        
        self.is_running = True
        start_time = time.time()
        cycles_run = 0
        converged = False
        
        logger.info(f"Iniciando loop contínuo: max_cycles={max_cycles}")
        
        try:
            while self.is_running and cycles_run < max_cycles:
                cycle = self.run_cycle()
                cycles_run += 1
                
                # Log periódico
                if cycles_run % self.config.log_every_n_cycles == 0:
                    logger.info(
                        f"Ciclo {cycles_run}/{max_cycles}: "
                        f"reward={cycle.avg_reward:.3f}, "
                        f"success_rate={cycle.success_rate:.1%}"
                    )
                
                # Salvar métricas periodicamente
                if cycles_run % self.config.save_metrics_every_n_cycles == 0:
                    self._save_metrics()
                
                # Verificar convergência
                if stop_on_convergence and self.metrics.is_converged():
                    converged = True
                    logger.info(f"Loop convergiu após {cycles_run} ciclos")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Loop interrompido pelo usuário")
        finally:
            self.is_running = False
        
        total_time = time.time() - start_time
        
        # Forçar aprendizado de dados restantes
        self.learner.force_learn()
        
        # Salvar métricas finais
        self._save_metrics()
        
        self.last_run_summary = {
            "cycles_run": cycles_run,
            "total_time_seconds": total_time,
            "converged": converged,
            "final_convergence_score": self.metrics.get_convergence_score(),
            "metrics_summary": self.metrics.get_summary(),
            "executor_stats": self.executor.get_stats(),
            "collector_stats": self.collector.get_stats(),
            "learner_stats": self.learner.get_stats()
        }
        
        return self.last_run_summary
    
    def stop(self):
        """Para o loop em execução"""
        self.is_running = False
        logger.info("Loop stop solicitado")
    
    def _detect_gaps(self) -> List[Dict[str, Any]]:
        """Detecta knowledge gaps via abduction_engine"""
        if not self.abduction_engine:
            # Retornar gaps simulados se não tem engine
            return [
                {"gap_id": "gap_1", "description": "Simulated gap", "source": "sim"}
            ]
        
        try:
            return self.abduction_engine.detect_knowledge_gaps()
        except Exception as e:
            logger.error(f"Erro ao detectar gaps: {e}")
            return []
    
    def _generate_hypotheses(self, gaps: List[Dict]) -> List[Dict[str, Any]]:
        """Gera hipóteses para cada gap"""
        if not self.abduction_engine:
            # Retornar hipóteses simuladas
            return [
                {
                    "id": f"hyp_{i}",
                    "hypothesis_text": f"Simulated hypothesis for {gap.get('gap_id', i)}",
                    "source_cluster": "cluster_a",
                    "target_cluster": "cluster_b",
                    "confidence_score": 0.5 + (i * 0.1),
                    "test_requirements": []
                }
                for i, gap in enumerate(gaps)
            ]
        
        hypotheses = []
        for gap in gaps:
            try:
                gap_hypotheses = self.abduction_engine.generate_hypotheses(gap)
                hypotheses.extend(gap_hypotheses)
            except Exception as e:
                logger.error(f"Erro ao gerar hipóteses para gap: {e}")
        
        return hypotheses
    
    def _save_metrics(self):
        """Salva métricas em arquivo"""
        try:
            self.metrics.save_to_file(self.config.metrics_save_path)
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do loop"""
        return {
            "is_running": self.is_running,
            "current_cycle": self.current_cycle,
            "metrics_summary": self.metrics.get_summary(),
            "last_run_summary": self.last_run_summary
        }
    
    def reset(self):
        """Reseta o estado do loop"""
        self.is_running = False
        self.current_cycle = 0
        self.last_run_summary = {}
        self.metrics.reset()
        self.executor.reset_stats()
        self.collector.reset_stats()
        self.learner.reset()
        logger.info("Loop resetado")
