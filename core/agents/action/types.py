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
