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
from .action_selection import ActionSelectionAdapter, LoopState, AgentAction

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
    # Active Inference modes
    use_active_inference_shadow: bool = False  # Log AI suggestions without changing behavior
    use_active_inference: bool = False          # Use AI as primary decision source


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
        on_action_complete: Optional[Callable] = None,
        active_inference_adapter: Optional[ActionSelectionAdapter] = None,
        mycelial=None,  # Optional MycelialReasoning for stats
        field=None      # Optional PreStructuralField for stats
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
        
        # Active Inference
        self.active_inference_adapter = active_inference_adapter
        self.shadow_actions: List[AgentAction] = []
        self.ai_primary_actions: List[AgentAction] = []  # Actions used when AI is primary
        self.ai_fallback_count: int = 0  # Count of times heuristic was used as fallback
        
        # External components for stats
        self.mycelial = mycelial  # MycelialReasoning instance
        self.field = field        # PreStructuralField instance
        
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
            
            # Shadow mode: call Active Inference once per cycle (logging only)
            if self.config.use_active_inference_shadow and not self.config.use_active_inference:
                if self.active_inference_adapter:
                    loop_state = self._build_loop_state(gaps, hypotheses, 0.0)
                    try:
                        ai_action = self.active_inference_adapter.select_action(loop_state)
                        self.shadow_actions.append(ai_action)
                        logger.info(
                            f"[SHADOW] ActiveInference: {ai_action.action_type.name} "
                            f"target={ai_action.target} EFE={ai_action.expected_free_energy:.3f}"
                        )
                    except Exception as e:
                        logger.warning(f"[SHADOW] ActiveInference failed: {e}")
            
            # Primary mode: use Active Inference as main decision source
            if self.config.use_active_inference and self.active_inference_adapter:
                hypotheses = self._get_ai_hypotheses(gaps, hypotheses)
            
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
        
        # AbductionEngine.generate_hypotheses() reads gaps from internal state
        # and returns Hypothesis objects - we need to call it once, not per-gap
        try:
            # generate_hypotheses reads from self.knowledge_gaps (populated by detect_knowledge_gaps)
            hypothesis_objs = self.abduction_engine.generate_hypotheses(max_hypotheses=min(len(gaps), 10))
            
            # Convert Hypothesis objects to dicts for executor
            hypotheses = []
            for h in hypothesis_objs:
                hypotheses.append({
                    "id": h.id,
                    "hypothesis_text": h.hypothesis_text,
                    "source_cluster": h.source_cluster,
                    "target_cluster": h.target_cluster,
                    "confidence_score": h.confidence_score,
                    "test_requirements": h.test_requirements
                })
            return hypotheses
        except Exception as e:
            logger.error(f"Erro ao gerar hipóteses: {e}")
            return []
    
    def _save_metrics(self):
        """Salva métricas em arquivo"""
        try:
            self.metrics.save_to_file(self.config.metrics_save_path)
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def _build_loop_state(
        self, 
        gaps: List[Dict], 
        hypotheses: List[Dict], 
        last_reward: float
    ) -> LoopState:
        """
        Build LoopState for Active Inference adapter.
        
        Extracts real stats from mycelial and field components if available.
        """
        return LoopState(
            cycle=self.current_cycle,
            gaps=gaps,
            hypotheses=hypotheses,
            mycelial_stats=self._get_mycelial_stats(),
            field_stats=self._get_field_stats(),
            last_reward=last_reward,
            recent_actions=self.shadow_actions[-5:] if self.shadow_actions else []
        )
    
    def _get_mycelial_stats(self) -> Dict[str, Any]:
        """
        Extract stats from MycelialReasoning if available.
        
        Returns empty dict if mycelial is not connected.
        """
        if not self.mycelial:
            return {}
        
        try:
            stats = self.mycelial.get_network_stats()
            return {
                "active_nodes": stats.get("active_nodes", 0),
                "active_edges": stats.get("active_edges", 0),
                "total_observations": stats.get("total_observations", 0),
                "mean_weight": stats.get("mean_weight", 0.0),
                "max_weight": stats.get("max_weight", 0.0),
            }
        except Exception as e:
            logger.debug(f"Could not get mycelial stats: {e}")
            return {}
    
    def _get_field_stats(self) -> Dict[str, Any]:
        """
        Extract stats from PreStructuralField if available.
        
        Returns empty dict if field is not connected.
        """
        if not self.field:
            return {}
        
        try:
            stats = {}
            
            # Get manifold stats
            if hasattr(self.field, 'manifold') and self.field.manifold:
                stats["manifold_points"] = len(self.field.manifold.points)
                stats["manifold_dim"] = getattr(self.field.manifold, 'current_dim', 0)
            
            # Get trigger count
            if hasattr(self.field, 'trigger_count'):
                stats["trigger_count"] = self.field.trigger_count
            
            # Get free energy if available
            if hasattr(self.field, 'free_energy') and self.field.free_energy:
                try:
                    if hasattr(self.field.free_energy, 'last_F'):
                        stats["last_free_energy"] = float(self.field.free_energy.last_F)
                except:
                    pass
            
            return stats
        except Exception as e:
            logger.debug(f"Could not get field stats: {e}")
            return {}
    
    def _get_ai_hypotheses(
        self, 
        gaps: List[Dict], 
        fallback_hypotheses: List[Dict]
    ) -> List[Dict]:
        """
        Get hypotheses from Active Inference agent (primary mode).
        
        Falls back to heuristic hypotheses if AI fails.
        
        Returns:
            List of hypothesis dicts ready for HypothesisExecutor
        """
        try:
            loop_state = self._build_loop_state(gaps, fallback_hypotheses, 0.0)
            ai_action = self.active_inference_adapter.select_action(loop_state)
            
            # Track AI action
            self.ai_primary_actions.append(ai_action)
            
            # Convert to hypothesis for executor
            hypothesis = self._action_to_hypothesis(ai_action)
            
            logger.info(
                f"[AI-PRIMARY] ActiveInference: {ai_action.action_type.name} "
                f"target={ai_action.target} EFE={ai_action.expected_free_energy:.3f}"
            )
            
            # Return AI-selected action as the only hypothesis
            return [hypothesis]
            
        except Exception as e:
            logger.warning(f"[AI-FALLBACK] Heuristic used due to: {e}")
            self.ai_fallback_count += 1
            return fallback_hypotheses
    
    def _action_to_hypothesis(self, agent_action: AgentAction) -> Dict[str, Any]:
        """
        Convert an AgentAction to a hypothesis dict for HypothesisExecutor.
        
        Convention:
        - id: "ai_{action_type}_{target_hash}"
        - hypothesis_text: describes the action
        - source_cluster: from action parameters or target
        - target_cluster: from action parameters or target  
        - confidence_score: from action.confidence
        - test_requirements: empty list
        - _source: "active_inference" (metadata field)
        
        Args:
            agent_action: AgentAction from adapter
            
        Returns:
            Dict compatible with HypothesisExecutor.execute()
        """
        # Extract source/target from parameters or use action target
        params = agent_action.parameters or {}
        source = params.get("source", params.get("concept1", agent_action.target))
        target = params.get("target", params.get("concept2", agent_action.target))
        
        return {
            "id": f"ai_{agent_action.action_type.name}_{hash(agent_action.target) % 10000}",
            "hypothesis_text": f"AI-selected action: {agent_action.action_type.name} on {agent_action.target}",
            "source_cluster": str(source),
            "target_cluster": str(target),
            "confidence_score": agent_action.confidence,
            "test_requirements": [],
            "_source": "active_inference",
            "_action_type": agent_action.action_type.name,
            "_expected_free_energy": agent_action.expected_free_energy
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do loop"""
        return {
            "is_running": self.is_running,
            "current_cycle": self.current_cycle,
            "metrics_summary": self.metrics.get_summary(),
            "last_run_summary": self.last_run_summary,
            "shadow_actions_count": len(self.shadow_actions),
            "ai_primary_actions_count": len(self.ai_primary_actions),
            "ai_fallback_count": self.ai_fallback_count
        }
    
    def reset(self):
        """Reseta o estado do loop"""
        self.is_running = False
        self.current_cycle = 0
        self.last_run_summary = {}
        self.shadow_actions = []
        self.ai_primary_actions = []
        self.ai_fallback_count = 0
        self.metrics.reset()
        self.executor.reset_stats()
        self.collector.reset_stats()
        self.learner.reset()
        logger.info("Loop resetado")
