"""
Alexandria - Action Agent
Main orchestrator for the Action Agent system.

This module coordinates all components (security, parameters, executors, testing, evidence)
to execute actions and test hypotheses.
"""

import json
import time
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict

from .types import (
    ActionType, ActionStatus, EvidenceType,
    ActionResult, TestHypothesis
)
from .security_controller import SecurityController
from .parameter_controller import ParameterController
from .execution import (
    execute_api_call,
    execute_model_retrain,
    execute_data_generation,
    execute_simulation,
    execute_internal_learning,
    execute_config_change,
    execute_parameter_adjustment,
)

from core.reasoning.neural_learner import V2Learner

logger = logging.getLogger(__name__)


class ActionAgent:
    """
    Action Agent - As "mãos" da ASI para executar ações no mundo real.
    
    Responsabilidades:
    - Execução segura de ações baseadas em hipóteses
    - Interface com APIs e sistemas externos
    - Registro de resultados como evidência
    - Controle de segurança e auditoria
    """
    
    def __init__(self, sfs_path: str = "./data/", security_level: str = "strict"):
        self.sfs_path = Path(sfs_path)
        self.security_controller = SecurityController()
        self.parameter_controller = ParameterController()
        self.action_results = {}
        self.test_hypotheses = {}
        self.security_level = security_level
        self.v2_learner = V2Learner()
        
        # Criar diretórios necessários
        self.sfs_path.mkdir(exist_ok=True)
        
        logger.info("Action Agent inicializado (refatorado)")
        logger.info(f"Security Level: {security_level}")
        logger.info(f"SFS Path: {self.sfs_path}")
    
    def execute_action(self, action_type: ActionType, parameters: Dict[str, Any]) -> ActionResult:
        """
        Executa uma ação do tipo especificado com os parâmetros fornecidos.
        
        Args:
            action_type: Tipo de ação a executar
            parameters: Parâmetros da ação
            
        Returns:
            ActionResult com o resultado da execução
        """
        action_id = self._generate_action_id()
        start_time = datetime.now()
        
        logger.info(f"Executando ação {action_type.value} (ID: {action_id})")
        
        # Verificar rate limiting
        if not self.security_controller.check_rate_limit(action_type):
            result = ActionResult(
                action_id=action_id,
                action_type=action_type,
                status=ActionStatus.FAILED,
                start_time=start_time,
                error_message="Rate limit excedido"
            )
            self.action_results[action_id] = result
            return result
        
        # Executar ação baseada no tipo
        try:
            if action_type == ActionType.API_CALL:
                result = execute_api_call(parameters, self.security_controller, action_id)
            elif action_type == ActionType.PARAMETER_ADJUSTMENT:
                result = execute_parameter_adjustment(parameters, self.parameter_controller, action_id)
            elif action_type == ActionType.MODEL_RETRAIN:
                result = execute_model_retrain(parameters, action_id)
            elif action_type == ActionType.DATA_GENERATION:
                result = execute_data_generation(parameters, self.sfs_path, action_id)
            elif action_type == ActionType.SYSTEM_CONFIG_CHANGE:
                result = execute_config_change(parameters, action_id)
            elif action_type == ActionType.SIMULATION_RUN:
                result = execute_simulation(parameters, action_id)
            elif action_type == ActionType.INTERNAL_LEARNING:
                result = execute_internal_learning(parameters, self.v2_learner, action_id)
            else:
                raise ValueError(f"Tipo de ação não suportado: {action_type}")
            
            result.end_time = datetime.now()
            
            # Log de auditoria
            self.security_controller.log_action(result, {
                "parameters": parameters,
                "result_data_keys": list(result.result_data.keys()) if result.result_data else [],
                "duration": result.duration
            })
            
            self.action_results[action_id] = result
            return result
            
        except Exception as e:
            result = ActionResult(
                action_id=action_id,
                action_type=action_type,
                status=ActionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
            
            self.action_results[action_id] = result
            return result
    
    def _generate_action_id(self) -> str:
        """Gera ID único para ação"""
        timestamp = str(int(time.time() * 1000))
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"ACT_{timestamp}_{random_suffix}"
    
    def test_hypothesis(self, hypothesis: Dict[str, Any]) -> TestHypothesis:
        """
        Testa uma hipótese gerada pelo V9 usando ações apropriadas.
        
        Args:
            hypothesis: Hipótese do V9 com informações de teste
            
        Returns:
            TestHypothesis com resultado do teste
        """
        hypothesis_id = hypothesis.get("id", f"HYP_{int(time.time())}")
        
        # Criar estrutura de teste
        test_hyp = TestHypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis.get("hypothesis_text", ""),
            source_cluster=hypothesis.get("source_cluster", 0),
            target_cluster=hypothesis.get("target_cluster", 1),
           test_action=ActionType(hypothesis.get("test_action", "simulation_run")),
            test_parameters=hypothesis.get("test_parameters", {}),
            expected_outcome=hypothesis.get("expected_outcome", {}),
            validation_criteria=hypothesis.get("validation_criteria", {}),
            created_at=datetime.now()
        )
        
        # Executar teste
        logger.info(f"Testando hipótese: {hypothesis_id}")
        
        try:
            # Executar ação de teste
            result = self.execute_action(
                action_type=test_hyp.test_action,
                parameters=test_hyp.test_parameters
            )
            
            test_hyp.result = result
            test_hyp.executed_at = datetime.now()
            
            # Determinar se a hipótese é suportada ou refutada
            evidence_type = self._evaluate_test_result(result, test_hyp)
            test_hyp.result.evidence_generated = True
            test_hyp.result.evidence_type = evidence_type
            
            # Registrar evidência no SFS
            self._register_test_evidence(test_hyp)
            
            logger.info(f"Teste de hipótese concluído: {hypothesis_id} - {evidence_type.value}")
            
        except Exception as e:
            logger.error(f"Erro no teste de hipótese {hypothesis_id}: {e}")
        
        self.test_hypotheses[hypothesis_id] = test_hyp
        return test_hyp
    
    def _evaluate_test_result(self, result: ActionResult, test_hyp: TestHypothesis) -> EvidenceType:
        """Avalia resultado do teste para determinar suporte/refutação"""
        try:
            if result.status != ActionStatus.COMPLETED:
                return EvidenceType.INCONCLUSIVE
            
            # Lógica específica baseada no tipo de ação
            if result.action_type == ActionType.PARAMETER_ADJUSTMENT:
                return self._evaluate_parameter_adjustment(result, test_hyp)
            elif result.action_type == ActionType.MODEL_RETRAIN:
                return self._evaluate_model_retrain(result, test_hyp)
            elif result.action_type == ActionType.SIMULATION_RUN:
                return self._evaluate_simulation_run(result, test_hyp)
            else:
                return EvidenceType.NEUTRAL
                
        except Exception as e:
            logger.error(f"Erro na avaliação do resultado: {e}")
            return EvidenceType.INCONCLUSIVE
    
    def _evaluate_parameter_adjustment(self, result: ActionResult, test_hyp: TestHypothesis) -> EvidenceType:
        """Avalia resultado de ajuste de parâmetro"""
        data = result.result_data or {}
        success = data.get("adjustment_success", False)
        
        if not success:
            return EvidenceType.CONTRADICTING
        
        expected = test_hyp.expected_outcome
        achieved = data.get("new_value")
        
        if expected.get("target_value") == achieved:
            return EvidenceType.SUPPORTING
        elif "range" in expected:
            target_range = expected["range"]
            if target_range[0] <= achieved <= target_range[1]:
                return EvidenceType.SUPPORTING
        
        return EvidenceType.NEUTRAL
    
    def _evaluate_model_retrain(self, result: ActionResult, test_hyp: TestHypothesis) -> EvidenceType:
        """Avalia resultado de re-treinamento de modelo"""
        data = result.result_data or {}
        convergence = data.get("convergence", False)
        accuracy = data.get("accuracy", 0)
        
        expected_accuracy = test_hyp.validation_criteria.get("min_accuracy", 0.7)
        
        if convergence and accuracy >= expected_accuracy:
            return EvidenceType.SUPPORTING
        elif not convergence:
            return EvidenceType.CONTRADICTING
        
        return EvidenceType.NEUTRAL
    
    def _evaluate_simulation_run(self, result: ActionResult, test_hyp: TestHypothesis) -> EvidenceType:
        """Avalia resultado de simulação"""
        data = result.result_data or {}
        metrics = data.get("metrics", {})
        
        min_convergence = test_hyp.validation_criteria.get("min_convergence", 0.5)
        min_stability = test_hyp.validation_criteria.get("min_stability", 0.7)
        
        convergence = metrics.get("convergence_rate", 0)
        stability = metrics.get("stability", 0)
        
        if convergence >= min_convergence and stability >= min_stability:
            return EvidenceType.SUPPORTING
        elif convergence < min_convergence * 0.5:
            return EvidenceType.CONTRADICTING
        
        return EvidenceType.NEUTRAL
    
    def _register_test_evidence(self, test_hyp: TestHypothesis):
        """Registra evidência do teste no SFS"""
        try:
            evidence_data = {
                "evidence_id": f"EV_{test_hyp.hypothesis_id}_{int(time.time())}",
                "hypothesis_id": test_hyp.hypothesis_id,
                "test_date": test_hyp.executed_at.isoformat(),
                "hypothesis_text": test_hyp.hypothesis_text,
                "test_result": test_hyp.result.status.value,
                "evidence_type": test_hyp.result.evidence_type.value if test_hyp.result.evidence_type else "neutral",
                "result_data": asdict(test_hyp.result) if test_hyp.result else {},
                "clusters": [test_hyp.source_cluster, test_hyp.target_cluster],
                "test_parameters": test_hyp.test_parameters
            }
            
            evidence_file = self.sfs_path / f"test_evidence_{test_hyp.hypothesis_id}.json"
            with open(evidence_file, 'w', encoding='utf-8') as f:
                json.dump(evidence_data, f, indent=2, ensure_ascii=False)
            
            test_hyp.evidence_registered = True
            logger.info(f"Evidência registrada: {evidence_file}")
            
        except Exception as e:
            logger.error(f"Erro ao registrar evidência: {e}")
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos testes executados"""
        total_tests = len(self.test_hypotheses)
        if total_tests == 0:
            return {"total_tests": 0}
        
        evidence_counts = {ev_type.value: 0 for ev_type in EvidenceType}
        evidence_counts["total"] = total_tests
        
        status_counts = {}
        
        for test in self.test_hypotheses.values():
            if test.result and test.result.evidence_type:
                evidence_counts[test.result.evidence_type.value] += 1
            
            if test.result:
                status = test.result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        durations = [
            (test.result.duration if test.result else 0)
            for test in self.test_hypotheses.values()
            if test.result
        ]
        
        stats = {
            "total_tests": total_tests,
            "evidence_distribution": evidence_counts,
            "status_distribution": status_counts,
            "average_duration": np.mean(durations) if durations else 0,
            "success_rate": status_counts.get("completed", 0) / total_tests if total_tests > 0 else 0
        }
        
        return stats


def create_action_agent_system(sfs_instance, sfs_path: str = "./data/"):
    """
    Cria e inicializa o sistema completo de Action Agent.
    
    Args:
        sfs_instance: Instância do SFS multi-modal
        sfs_path: Caminho para dados do SFS
        
    Returns:
        Tupla com (ActionAgent, TestSimulator, EvidenceRegistrar)
    """
    from .test_simulator import TestSimulator
    from .evidence_registrar import EvidenceRegistrar
    
    # Criar Action Agent
    action_agent = ActionAgent(sfs_path=sfs_path)
    
    # Criar Test Simulator
    test_simulator = TestSimulator(action_agent)
    
    # Criar Evidence Registrar
    evidence_registrar = EvidenceRegistrar(action_agent, sfs_instance)
    
    logger.info("Sistema Action Agent inicializado com sucesso (refatorado)")
    return action_agent, test_simulator, evidence_registrar
