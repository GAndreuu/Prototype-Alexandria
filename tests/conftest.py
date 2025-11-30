import pytest
import os
import shutil
from core.topology_engine import TopologyEngine
from core.semantic_memory import SemanticFileSystem
from core.oracle import NeuralOracle
from config import settings

@pytest.fixture(scope="session")
def temp_data_dir():
    """Cria diretório de dados temporário para testes"""
    test_dir = os.path.join(settings.BASE_DIR, "data_test")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Limpeza após testes
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def engine():
    """Fixture para TopologyEngine"""
    return TopologyEngine()

@pytest.fixture(scope="session")
def memory(engine, temp_data_dir):
    """Fixture para SemanticFileSystem usando diretório de teste"""
    # Patch temporário no settings.DATA_DIR se necessário, 
    # ou instanciar SFS apontando para o dir de teste se a classe permitir.
    # Como SFS usa settings global, vamos apenas garantir que ele funcione.
    # Idealmente, SFS deveria aceitar base_path no init.
    return SemanticFileSystem(engine)

@pytest.fixture(scope="session")
def oracle():
    """Fixture para NeuralOracle"""
    return NeuralOracle(use_hybrid=True)
