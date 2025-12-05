"""
Test compatibility and functionality of refactored Action Agent.

This module tests:
1. Backward compatibility (old imports still work)
2. New structure imports work correctly
3. Functional tests (actions execute properly)
"""

import pytest
import warnings


def test_backward_compatibility():
    """Testa que imports antigos ainda funcionam (com warning)"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        from core.agents.action_agent import ActionAgent, ActionType
        
        # Verificar que warning foi emitido
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message).lower()
        
        # Verificar que classes foram importadas
        assert ActionAgent is not None
        assert ActionType is not None


def test_new_structure_imports():
    """Testa que nova estrutura funciona"""
    from core.agents.action import (
        ActionAgent, ActionType, ActionStatus, EvidenceType,
        SecurityController, ParameterController,
        TestSimulator, EvidenceRegistrar,
        create_action_agent_system
    )
    
    # Verificar que todas as classes foram importadas
    assert all([
        ActionAgent, ActionType, ActionStatus, EvidenceType,
        SecurityController, ParameterController,
        TestSimulator, EvidenceRegistrar,
        create_action_agent_system
    ])


def test_security_controller_standalone():
    """Testa SecurityController isoladamente"""
    from core.agents.action.security_controller import SecurityController
    
    controller = SecurityController()
    assert controller is not None
    assert hasattr(controller, 'validate_api_call')
    assert hasattr(controller, 'check_rate_limit')
    assert hasattr(controller, 'log_action')
    assert hasattr(controller, 'get_audit_log')


def test_parameter_controller_standalone():
    """Testa ParameterController isoladamente"""
    from core.agents.action.parameter_controller import ParameterController
    
    controller = ParameterController()
    assert controller is not None
    assert hasattr(controller, 'adjust_parameter')
    assert hasattr(controller, 'reset_parameter')
    assert hasattr(controller, 'get_parameter')
    assert hasattr(controller, 'get_parameter_history')
    
    # Testar ajuste de parâmetro
    result = controller.adjust_parameter("V11_BETA", 2.5)
    assert result is True
    assert controller.get_parameter("V11_BETA") == 2.5


def test_action_agent_creation():
    """Testa criação do ActionAgent"""
    from core.agents.action import ActionAgent
    
    agent = ActionAgent(sfs_path="./data/test")
    assert agent is not None
    assert hasattr(agent, 'execute_action')
    assert hasattr(agent, 'test_hypothesis')
    assert hasattr(agent, 'get_test_statistics')


def test_functional_action_agent():
    """Teste funcional completo do Action Agent"""
    from core.agents.action import create_action_agent_system, ActionType, ActionStatus
    
    # Mock SFS
    class MockSFS:
        def index_file(self, file_path):
            return 1
    
    mock_sfs = MockSFS()
    action_agent, test_simulator, evidence_registrar = create_action_agent_system(mock_sfs, sfs_path="./data/test")
    
    # Teste 1: Ajuste de parâmetro
    result = action_agent.execute_action(
        action_type=ActionType.PARAMETER_ADJUSTMENT,
        parameters={"parameter_name": "V11_BETA", "new_value": 2.5}
    )
    assert result.status == ActionStatus.COMPLETED
    assert result.result_data["adjustment_success"] is True
    
    # Teste 2: Geração de dados
    data_result = action_agent.execute_action(
        action_type=ActionType.DATA_GENERATION,
        parameters={"data_type": "random", "size": 100, "dimensions": 384}
    )
    assert data_result.status == ActionStatus.COMPLETED
    assert "data_file" in data_result.result_data
    
    # Teste 3: Simulação
    sim_result = action_agent.execute_action(
        action_type=ActionType.SIMULATION_RUN,
        parameters={"simulation_name": "test_sim", "duration": 1.0}
    )
    assert sim_result.status == ActionStatus.COMPLETED
    assert "metrics" in sim_result.result_data


def test_executor_modules():
    """Testa que módulos de execução podem ser importados"""
    from core.agents.action.execution import (
        execute_api_call,
        execute_model_retrain,
        execute_data_generation,
        execute_simulation,
        execute_internal_learning,
        execute_config_change,
        execute_parameter_adjustment
    )
    
    # Verificar que todos os executores foram importados
    assert all([
        execute_api_call,
        execute_model_retrain,
        execute_data_generation,
        execute_simulation,
        execute_internal_learning,
        execute_config_change,
        execute_parameter_adjustment
    ])


if __name__ == "__main__":
    print("=== Testes de Refatoração do Action Agent ===\n")
    
    print("1. Testando compatibilidade reversa...")
    test_backward_compatibility()
    print("   ✅ Compatibilidade reversa OK\n")
    
    print("2. Testando nova estrutura...")
    test_new_structure_imports()
    print("   ✅ Nova estrutura OK\n")
    
    print("3. Testando SecurityController...")
    test_security_controller_standalone()
    print("   ✅ SecurityController OK\n")
    
    print("4. Testando ParameterController...")
    test_parameter_controller_standalone()
    print("   ✅ ParameterController OK\n")
    
    print("5. Testando criação do ActionAgent...")
    test_action_agent_creation()
    print("   ✅ ActionAgent criado OK\n")
    
    print("6. Testando funcionalidade completa...")
    test_functional_action_agent()
    print("   ✅ Testes funcionais OK\n")
    
    print("7. Testando módulos de execução...")
    test_executor_modules()
    print("   ✅ Executores OK\n")
    
    print("=" * 50)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 50)
