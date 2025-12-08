"""
Loop Metrics - Tracking de performance do ciclo
=================================================

Métricas por ciclo e agregadas para monitorar o self-feeding loop
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Métricas de um único ciclo"""
    cycle_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Detecção
    gaps_detected: int = 0
    hypotheses_generated: int = 0
    
    # Execução
    actions_executed: int = 0
    actions_successful: int = 0
    
    # Feedback
    total_evidence: int = 0
    new_connections: int = 0
    avg_reward: float = 0.0
    
    # Aprendizado
    learning_triggered: bool = False
    loss: float = 0.0
    
    # Tempo
    cycle_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.actions_executed == 0:
            return 0.0
        return self.actions_successful / self.actions_executed
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        return d


class LoopMetrics:
    """
    Gerencia métricas do Self-Feeding Loop.
    
    Tracka:
    - Métricas por ciclo
    - Agregados totais
    - Detecção de convergência
    """
    
    def __init__(
        self,
        convergence_window: int = 10,
        convergence_threshold: float = 0.01
    ):
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        # Histórico de ciclos
        self.cycles: List[CycleMetrics] = []
        
        # Agregados
        self.total_cycles: int = 0
        self.total_gaps: int = 0
        self.total_hypotheses: int = 0
        self.total_actions: int = 0
        self.total_successful: int = 0
        self.total_evidence: int = 0
        self.total_connections: int = 0
        self.total_learning_events: int = 0
        self.cumulative_reward: float = 0.0
        
    def start_cycle(self) -> CycleMetrics:
        """Inicia um novo ciclo e retorna objeto de métricas"""
        cycle = CycleMetrics(
            cycle_id=self.total_cycles,
            timestamp=datetime.now().isoformat()
        )
        return cycle
    
    def record_cycle(self, cycle: CycleMetrics):
        """
        Registra um ciclo completo.
        
        Args:
            cycle: CycleMetrics preenchido com dados do ciclo
        """
        self.cycles.append(cycle)
        self.total_cycles += 1
        
        # Atualizar agregados
        self.total_gaps += cycle.gaps_detected
        self.total_hypotheses += cycle.hypotheses_generated
        self.total_actions += cycle.actions_executed
        self.total_successful += cycle.actions_successful
        self.total_evidence += cycle.total_evidence
        self.total_connections += cycle.new_connections
        self.cumulative_reward += cycle.avg_reward
        
        if cycle.learning_triggered:
            self.total_learning_events += 1
        
        logger.info(
            f"Ciclo {cycle.cycle_id} registrado: "
            f"gaps={cycle.gaps_detected}, "
            f"actions={cycle.actions_executed}, "
            f"success_rate={cycle.success_rate:.1%}"
        )
    
    def is_converged(self, threshold: Optional[float] = None) -> bool:
        """
        Verifica se o loop convergiu.
        
        Convergência = loss estável nas últimas N iterações
        """
        if len(self.cycles) < self.convergence_window:
            return False
        
        threshold = threshold or self.convergence_threshold
        
        recent = self.cycles[-self.convergence_window:]
        losses = [c.loss for c in recent if c.learning_triggered]
        
        if len(losses) < 2:
            return False
        
        # Variância do loss
        mean_loss = sum(losses) / len(losses)
        variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        
        return variance < threshold
    
    def get_convergence_score(self) -> float:
        """
        Retorna score de convergência (0-1, 1 = convergiu).
        """
        if len(self.cycles) < self.convergence_window:
            return 0.0
        
        recent = self.cycles[-self.convergence_window:]
        losses = [c.loss for c in recent if c.learning_triggered]
        
        if len(losses) < 2:
            return 0.0
        
        mean_loss = sum(losses) / len(losses)
        variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        
        # Score inversamente proporcional à variância
        score = 1.0 / (1.0 + variance * 100)
        return min(score, 1.0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas"""
        return {
            "total_cycles": self.total_cycles,
            "total_gaps": self.total_gaps,
            "total_hypotheses": self.total_hypotheses,
            "total_actions": self.total_actions,
            "success_rate": self.total_successful / self.total_actions if self.total_actions > 0 else 0,
            "total_evidence": self.total_evidence,
            "total_connections": self.total_connections,
            "total_learning_events": self.total_learning_events,
            "cumulative_reward": self.cumulative_reward,
            "avg_reward_per_cycle": self.cumulative_reward / self.total_cycles if self.total_cycles > 0 else 0,
            "convergence_score": self.get_convergence_score(),
            "is_converged": self.is_converged()
        }
    
    def get_recent_cycles(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retorna os N ciclos mais recentes"""
        return [c.to_dict() for c in self.cycles[-n:]]
    
    def get_trend(self, metric: str = "avg_reward", window: int = 10) -> float:
        """
        Calcula tendência de uma métrica.
        
        Returns:
            Valor positivo = melhoria, negativo = piora
        """
        if len(self.cycles) < window * 2:
            return 0.0
        
        # Comparar duas janelas
        old_window = self.cycles[-(window*2):-window]
        new_window = self.cycles[-window:]
        
        def get_metric(cycle, name):
            return getattr(cycle, name, 0)
        
        old_avg = sum(get_metric(c, metric) for c in old_window) / len(old_window)
        new_avg = sum(get_metric(c, metric) for c in new_window) / len(new_window)
        
        if old_avg == 0:
            return 0.0
        
        return (new_avg - old_avg) / abs(old_avg)
    
    def to_json(self) -> str:
        """Exporta todas as métricas para JSON"""
        return json.dumps({
            "summary": self.get_summary(),
            "cycles": [c.to_dict() for c in self.cycles]
        }, indent=2)
    
    def save_to_file(self, filepath: str):
        """Salva métricas em arquivo JSON"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Métricas salvas em: {filepath}")
    
    def reset(self):
        """Reseta todas as métricas"""
        self.cycles = []
        self.total_cycles = 0
        self.total_gaps = 0
        self.total_hypotheses = 0
        self.total_actions = 0
        self.total_successful = 0
        self.total_evidence = 0
        self.total_connections = 0
        self.total_learning_events = 0
        self.cumulative_reward = 0.0
