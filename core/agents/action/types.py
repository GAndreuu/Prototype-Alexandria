"""
Alexandria - Action Agent Types
Type definitions for the Action Agent system.

This module contains all enums and dataclasses used across the Action Agent.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


class ActionType(Enum):
    """Tipos de ações suportadas pelo Action Agent"""
    API_CALL = "api_call"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    MODEL_RETRAIN = "model_retrain"
    DATA_GENERATION = "data_generation"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SIMULATION_RUN = "simulation_run"
    INTERNAL_LEARNING = "internal_learning"


class ActionStatus(Enum):
    """Status de execução de ações"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceType(Enum):
    """Tipos de evidência gerada pelos testes"""
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ActionResult:
    """Resultado de uma ação executada"""
    action_id: str
    action_type: ActionType
    status: ActionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    evidence_generated: bool = False
    evidence_type: Optional[EvidenceType] = None
    
    @property
    def duration(self) -> float:
        """Duração da execução em segundos"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa o resultado para dicionário (compatível com serialização legada)"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value if hasattr(self.action_type, 'value') else str(self.action_type),
            "status": self.status.value if hasattr(self.status, 'value') else str(self.status),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "evidence_generated": self.evidence_generated,
            "evidence_type": self.evidence_type.value if self.evidence_type and hasattr(self.evidence_type, 'value') else str(self.evidence_type) if self.evidence_type else None,
            "duration": self.duration
        }


@dataclass
class TestHypothesis:
    """Estrutura para teste de hipóteses"""
    hypothesis_id: str
    hypothesis_text: str
    source_cluster: int
    target_cluster: int
    test_action: ActionType
    test_parameters: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[ActionResult] = None
    evidence_registered: bool = False
