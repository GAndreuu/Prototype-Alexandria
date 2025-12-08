"""
Feedback Collector - Coleta feedback das ações para treino
============================================================

Conecta action_result → treinamento neural
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingFeedback:
    """Feedback formatado para treinamento neural"""
    embeddings: List[np.ndarray] = field(default_factory=list)
    reward_signal: float = 0.0
    should_learn: bool = False
    source_action_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings,  # Passar embeddings reais
            "embeddings_count": len(self.embeddings),
            "reward_signal": self.reward_signal,
            "should_learn": self.should_learn,
            "source_action_type": self.source_action_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ActionFeedbackCollector:
    """
    Coleta e processa feedback das ações executadas.
    
    Converte resultados de ação em formato para treinamento neural:
    - Extrai embeddings das evidências encontradas
    - Calcula reward signal baseado no sucesso
    - Determina se deve triggar aprendizado
    
    Regras de Reward:
    - success=True e evidence > 0: reward = +0.5 a +1.0
    - success=True mas evidence = 0: reward = 0
    - success=False: reward = -0.5
    - new_connections > 0: bonus +0.3
    """
    
    def __init__(
        self,
        topology_engine=None,
        learning_threshold: float = 0.3,
        max_reward: float = 1.0,
        min_reward: float = -0.5,
        connection_bonus: float = 0.3
    ):
        self.topology_engine = topology_engine
        self.learning_threshold = learning_threshold
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.connection_bonus = connection_bonus
        
        # Histórico
        self.feedback_history: List[TrainingFeedback] = []
        self.total_collected = 0
        self.total_positive = 0
        self.total_negative = 0
        
        logger.info("ActionFeedbackCollector inicializado")
    
    def collect(self, action_result: Dict[str, Any]) -> TrainingFeedback:
        """
        Coleta feedback de um resultado de ação.
        
        Args:
            action_result: Dict com keys:
                - action: dict (action executada)
                - success: bool
                - evidence_found: List[str]
                - new_connections: int
                - execution_time_ms: float
                - error_message: str (opcional)
                - metadata: dict (opcional)
        
        Returns:
            TrainingFeedback formatado para neural learner
        """
        success = action_result.get("success", False)
        evidence = action_result.get("evidence_found", [])
        new_connections = action_result.get("new_connections", 0)
        action = action_result.get("action", {})
        action_type = action.get("action_type", "UNKNOWN")
        
        # Calcular reward
        reward = self._calculate_reward(success, evidence, new_connections)
        
        # Extrair embeddings das evidências
        embeddings = self._extract_embeddings(evidence)
        
        # Determinar se deve aprender
        should_learn = abs(reward) >= self.learning_threshold
        
        # Criar feedback
        feedback = TrainingFeedback(
            embeddings=embeddings,
            reward_signal=reward,
            should_learn=should_learn,
            source_action_type=action_type,
            metadata={
                "evidence_count": len(evidence),
                "new_connections": new_connections,
                "execution_time_ms": action_result.get("execution_time_ms", 0),
                "success": success
            }
        )
        
        # Atualizar estatísticas
        self.total_collected += 1
        if reward > 0:
            self.total_positive += 1
        elif reward < 0:
            self.total_negative += 1
            
        # Salvar no histórico
        self.feedback_history.append(feedback)
        
        logger.debug(
            f"Feedback coletado: reward={reward:.2f}, "
            f"should_learn={should_learn}, "
            f"embeddings={len(embeddings)}"
        )
        
        return feedback
    
    def _calculate_reward(
        self, 
        success: bool, 
        evidence: List[str], 
        new_connections: int
    ) -> float:
        """
        Calcula reward signal baseado no resultado da ação.
        """
        if not success:
            return self.min_reward
        
        if len(evidence) == 0 and new_connections == 0:
            return 0.0
        
        # Base reward proporcional à evidência encontrada
        evidence_score = min(len(evidence) / 5.0, 1.0)  # Normaliza para 0-1
        base_reward = 0.5 + (evidence_score * 0.5)  # 0.5 a 1.0
        
        # Bonus por novas conexões
        if new_connections > 0:
            base_reward += self.connection_bonus
        
        # Limitar ao máximo
        return min(base_reward, self.max_reward)
    
    def _extract_embeddings(self, evidence: List[str]) -> List[np.ndarray]:
        """
        Extrai embeddings das evidências usando topology_engine.
        """
        if not self.topology_engine or len(evidence) == 0:
            return []
        
        # Filtrar evidências válidas
        valid_evidence = [text for text in evidence if text and len(text.strip()) > 0]
        
        if not valid_evidence:
            return []
        
        try:
            # TopologyEngine.encode espera List[str] e retorna array (N, 384)
            embeddings_array = self.topology_engine.encode(valid_evidence)
            # Converter para lista de arrays individuais
            return [embeddings_array[i] for i in range(len(embeddings_array))]
        except Exception as e:
            logger.error(f"Erro ao extrair embeddings: {e}")
            return []
    
    def get_recent_feedback(self, n: int = 10) -> List[TrainingFeedback]:
        """Retorna os N feedbacks mais recentes"""
        return self.feedback_history[-n:]
    
    def get_avg_reward(self, window: int = 100) -> float:
        """Calcula reward médio dos últimos N feedbacks"""
        recent = self.feedback_history[-window:]
        if not recent:
            return 0.0
        return sum(f.reward_signal for f in recent) / len(recent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de coleta"""
        return {
            "total_collected": self.total_collected,
            "total_positive": self.total_positive,
            "total_negative": self.total_negative,
            "positive_rate": self.total_positive / self.total_collected if self.total_collected > 0 else 0,
            "avg_reward": self.get_avg_reward(),
            "history_size": len(self.feedback_history)
        }
    
    def clear_history(self):
        """Limpa histórico de feedback"""
        self.feedback_history = []
        
    def reset_stats(self):
        """Reseta todas as estatísticas"""
        self.feedback_history = []
        self.total_collected = 0
        self.total_positive = 0
        self.total_negative = 0
