
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

@pytest.fixture(scope="package", autouse=True)
def mock_heavy_dependencies():
    """
    Globally mock heavy dependencies for ALL unit tests.
    This ensures no unit test accidentally loads a 500MB model.
    """
    print("\n[DEBUG] APPLYING GLOBAL MOCKS")
    with patch("core.topology.topology_engine.SentenceTransformer") as MockST, \
         patch("core.memory.storage.lancedb") as MockDB, \
         patch("core.reasoning.neural_learner.V2Learner") as MockLearner, \
         patch("core.reasoning.vqvae.model.MonolithV13") as MockV13, \
         patch("core.reasoning.symbol_grounding.SymbolGrounder") as MockGrounder:
        
        # Mock SentenceTransformer
        mock_st = MockST.return_value
        mock_st.encode.side_effect = lambda texts, **kwargs: np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)
        
        # Mock LanceDB
        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value.to_pandas.return_value = []
        MockDB.connect.return_value.open_table.return_value = mock_table
        
        # Mock V2Learner
        mock_learner = MockLearner.return_value
        mock_learner.learn.return_value = {"total_loss": 0.1}
        mock_learner.encode.return_value = np.zeros((1, 10))
        
        yield
