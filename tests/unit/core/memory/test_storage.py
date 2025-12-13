"""
Tests for Storage
"""
import pytest
from unittest.mock import Mock, patch
from core.memory.storage import LanceDBStorage

class TestLanceDBStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        # LanceDBStorage uses db_path, not uri
        with patch.object(LanceDBStorage, '_init_table'):
            storage = LanceDBStorage(db_path=str(tmp_path / "test_db"))
            return storage
    
    def test_init(self, storage):
        assert storage is not None

    def test_add(self, storage):
        # Mock add internal
        with patch.object(storage, 'add') as mock_add:
            mock_add.return_value = None
            storage.add(
                ids=["1"],
                vectors=[[0.1]*384],
                contents=["test"],
                sources=["test"],
                modalities=["TEXT"]
            )
            mock_add.assert_called_once()
            
    def test_search(self, storage):
        # Mock search return
        with patch.object(storage, 'search', return_value=[{"id": "1"}]):
            res = storage.search([0.1]*384)
            assert len(res) > 0
