import pytest
import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Ensure project root is in path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.topology.topology_engine import TopologyEngine
from core.memory.semantic_memory import SemanticFileSystem
from core.agents.oracle import NeuralOracle
from config import settings

# =============================================================================
# GLOBAL FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def temp_data_dir():
    """Creates a temporary data directory for tests."""
    test_dir = ROOT_DIR / "data_test_env"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    # Patch settings to point to test dir
    # Note: This might need more specific patching depending on how settings are used
    original_data_dir = settings.DATA_DIR
    settings.DATA_DIR = str(test_dir)
    
    yield test_dir
    
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)
    settings.DATA_DIR = original_data_dir

@pytest.fixture(scope="session")
def mock_lancedb():
    """Mocks LanceDB connection returning valid DataFrames."""
    with patch("core.memory.storage.lancedb") as mock_db:
        mock_table = MagicMock()
        
        # Mock search result to return a DataFrame with expected columns
        # This prevents "KeyError: vector" or similar in code consuming results
        mock_df = pd.DataFrame({
            'vector': [np.random.rand(384) for _ in range(2)],
            'text': ['Mock Doc 1', 'Mock Doc 2'],
            'id': ['1', '2'],
            'source': ['mock_src_1', 'mock_src_2'],
            'timestamp': [1234567890, 1234567891]
        })
        
        mock_table.search.return_value.limit.return_value.to_pandas.return_value = mock_df
        mock_table.add.return_value = None
        
        mock_db.connect.return_value.open_table.return_value = mock_table
        yield mock_db

@pytest.fixture(scope="session")
def mock_filesystem():
    """Mocks file I/O operations."""
    with patch("builtins.open", mock_open(read_data="Mock File Content")) as mock_file:
        yield mock_file

@pytest.fixture(scope="session")
def mock_gemini():
    """Mocks Google Gemini API."""
    with patch("google.generativeai.GenerativeModel") as mock_model:
        mock_chat = MagicMock()
        mock_chat.send_message.return_value.text = "Mocked Gemini Response"
        mock_model.return_value.start_chat.return_value = mock_chat
        yield mock_model

# =============================================================================
# CORE COMPONENTS (Unit/Mocked)
# =============================================================================

@pytest.fixture(scope="function")
def unit_topology_engine():
    """Lightweight TopologyEngine for unit tests (mocked models)."""
    # Patch SentenceTransformer to avoid loading 500MB model
    with patch("core.topology.topology_engine.SentenceTransformer") as MockModel:
        # Mock encode to return random vectors of correct shape
        mock_instance = MockModel.return_value
        mock_instance.encode.return_value = np.random.rand(1, 384)
        
        engine = TopologyEngine() 
        # Force is_trained to true to avoid retraining logic in some paths
        engine.is_trained = True
        return engine

@pytest.fixture(scope="function")
def unit_memory(unit_topology_engine, temp_data_dir, mock_lancedb):
    """SemanticMemory using mocked DB."""
    return SemanticFileSystem(unit_topology_engine)

@pytest.fixture(scope="function")
def unit_oracle(mock_gemini):
    """NeuralOracle with mocked Gemini."""
    return NeuralOracle(use_hybrid=True)

# =============================================================================
# INTEGRATION COMPONENTS (Real I/O)
# =============================================================================

@pytest.fixture(scope="module")
def integration_topology():
    """Real TopologyEngine for integration tests."""
    return TopologyEngine()

@pytest.fixture(scope="module")
def integration_memory(integration_topology, temp_data_dir):
    """Real SemanticMemory (no mocks) writing to temp dir."""
    return SemanticFileSystem(integration_topology)
