"""
Incremental Learner - Aprendizado incremental do V2
====================================================

Acumula feedback e dispara treinamento em batches
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LearningSession:
    """Registro de uma sessão de aprendizado"""
    timestamp: str
    batch_size: int
    total_loss: float
    recon_loss: float
    vq_loss: float
    avg_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "batch_size": self.batch_size,
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "vq_loss": self.vq_loss,
            "avg_reward": self.avg_reward
        }


class IncrementalLearner:
    """
    Gerencia aprendizado incremental do modelo VQ-VAE.
    
    Acumula feedback em batches e dispara treinamento quando:
    - Batch atinge tamanho mínimo (batch_threshold)
    - force_learn() é chamado
    - Reward acumulado excede threshold
    
    Integra com V2Learner para treinamento real.
    """
    
    def __init__(
        self,
        v2_learner=None,
        batch_threshold: int = 10,
        min_embeddings_per_feedback: int = 1,
        reward_threshold: float = 3.0,
        auto_save: bool = True
    ):
        self.v2_learner = v2_learner
        self.batch_threshold = batch_threshold
        self.min_embeddings = min_embeddings_per_feedback
        self.reward_threshold = reward_threshold
        self.auto_save = auto_save
        
        # Estado do batch atual
        self.current_batch: List[np.ndarray] = []
        self.current_rewards: List[float] = []
        self.accumulated_reward: float = 0.0
        
        # Histórico
        self.learning_sessions: List[LearningSession] = []
        self.total_learned: int = 0
        self.last_loss: float = 0.0
        
        logger.info(
            f"IncrementalLearner inicializado: "
            f"batch_threshold={batch_threshold}"
        )
    
    def add_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        Adiciona feedback ao batch de aprendizado.
        
        Args:
            feedback: Dict com keys:
                - embeddings: List[np.ndarray] (vetores 384D)
                - reward_signal: float (-1 a 1)
                - should_learn: bool
        
        Returns:
            True se aprendizado foi triggado, False caso contrário
        """
        embeddings = feedback.get("embeddings", [])
        reward = feedback.get("reward_signal", 0.0)
        should_learn = feedback.get("should_learn", False)
        
        # Só adicionar se tem embeddings suficientes e deve aprender
        if should_learn and len(embeddings) >= self.min_embeddings:
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    self.current_batch.append(emb)
            
            self.current_rewards.append(reward)
            self.accumulated_reward += abs(reward)
            
            logger.debug(
                f"Feedback adicionado: +{len(embeddings)} embeddings, "
                f"batch_size={len(self.current_batch)}"
            )
        
        # Verificar se deve aprender
        should_trigger = (
            len(self.current_batch) >= self.batch_threshold or
            self.accumulated_reward >= self.reward_threshold
        )
        
        if should_trigger:
            return self._trigger_learning()
        
        return False
    
    def force_learn(self) -> Dict[str, Any]:
        """
        Força aprendizado mesmo com batch pequeno.
        
        Returns:
            Dict com métricas do aprendizado ou vazio se não havia dados
        """
        if len(self.current_batch) == 0:
            logger.warning("force_learn chamado mas batch está vazio")
            return {}
        
        self._trigger_learning()
        
        if self.learning_sessions:
            return self.learning_sessions[-1].to_dict()
        return {}
    
    def _trigger_learning(self) -> bool:
        """
        Executa o aprendizado com o batch atual.
        
        Returns:
            True se aprendizado foi bem sucedido
        """
        if len(self.current_batch) == 0:
            return False
        
        batch_size = len(self.current_batch)
        avg_reward = sum(self.current_rewards) / len(self.current_rewards) if self.current_rewards else 0.0
        
        logger.info(
            f"Triggando aprendizado: batch_size={batch_size}, "
            f"avg_reward={avg_reward:.3f}"
        )
        
        try:
            # Converter batch para lista de listas (formato do V2Learner)
            vectors = [emb.tolist() for emb in self.current_batch]
            
            # Chamar V2Learner
            if self.v2_learner:
                metrics = self.v2_learner.learn(vectors)
            else:
                # Simulação se não tem learner
                metrics = {
                    "total_loss": 0.01,
                    "recon_loss": 0.005,
                    "vq_loss": 0.003,
                    "ortho_loss": 0.002
                }
            
            # Registrar sessão
            session = LearningSession(
                timestamp=datetime.now().isoformat(),
                batch_size=batch_size,
                total_loss=metrics.get("total_loss", 0),
                recon_loss=metrics.get("recon_loss", 0),
                vq_loss=metrics.get("vq_loss", 0),
                avg_reward=avg_reward
            )
            self.learning_sessions.append(session)
            
            # Atualizar estatísticas
            self.total_learned += batch_size
            self.last_loss = metrics.get("total_loss", 0)
            
            # Auto-save modelo
            if self.auto_save and self.v2_learner:
                self.v2_learner.save_model()
            
            logger.info(
                f"Aprendizado concluído: loss={self.last_loss:.4f}, "
                f"total_learned={self.total_learned}"
            )
            
            # Limpar batch
            self._clear_batch()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no aprendizado: {e}")
            self._clear_batch()
            return False
    
    def _clear_batch(self):
        """Limpa o batch atual"""
        self.current_batch = []
        self.current_rewards = []
        self.accumulated_reward = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado"""
        return {
            "total_learned": self.total_learned,
            "current_batch_size": len(self.current_batch),
            "accumulated_reward": self.accumulated_reward,
            "last_loss": self.last_loss,
            "sessions_count": len(self.learning_sessions),
            "avg_loss": self._get_avg_loss()
        }
    
    def _get_avg_loss(self, window: int = 10) -> float:
        """Calcula loss médio das últimas N sessões"""
        recent = self.learning_sessions[-window:]
        if not recent:
            return 0.0
        return sum(s.total_loss for s in recent) / len(recent)
    
    def get_learning_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de sessões de aprendizado"""
        return [s.to_dict() for s in self.learning_sessions]
    
    def is_batch_ready(self) -> bool:
        """Verifica se batch está pronto para aprender"""
        return len(self.current_batch) >= self.batch_threshold
    
    def reset(self):
        """Reseta todo o estado"""
        self._clear_batch()
        self.learning_sessions = []
        self.total_learned = 0
        self.last_loss = 0.0
