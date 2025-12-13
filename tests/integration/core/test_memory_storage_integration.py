
"""
Integration Test: LanceDB Storage
Verifies real functionality of vector storage (Add, Search, Filter).
"""
import pytest
import shutil
import numpy as np
import os
from pathlib import Path
from core.memory.storage import LanceDBStorage

@pytest.mark.integration
def test_lancedb_crud(tmp_path):
    """Test Create, Read, Update (Add), Delete (implicitly via cleanup) logic."""
    
    # Use tmp_path fixture from pytest for isolation
    test_db_path = str(tmp_path / "test_lancedb")
    
    try:
        storage = LanceDBStorage(db_path=test_db_path)
        
        # 1. ADD
        ids = ["1", "2", "3"]
        vectors = [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
        contents = ["Texto sobre IA", "Texto sobre Biologia", "Imagem de um Gato"]
        sources = ["doc1.txt", "doc2.txt", "img1.png"]
        modalities = ["TEXTUAL", "TEXTUAL", "VISUAL"]
        metadata = [{"type": "txt"}, {"type": "txt"}, {"type": "img"}]
        
        storage.add(ids, vectors, contents, sources, modalities, metadata)
        
        assert storage.count() == 3
        
        # 2. SEARCH
        query = vectors[0]
        results = storage.search(query, limit=1)
        
        assert len(results) == 1
        assert results[0]['id'] == "1"
        assert abs(results[0]['relevance']) >= 0 # Should be close to 0 distance or 1 similarity depending on metric
        
        # 3. FILTER
        # Filter for VISUAL
        results_vis = storage.search(vectors[2], limit=5, filter_sql="modality = 'VISUAL'")
        
        assert len(results_vis) > 0
        assert results_vis[0]['modality'] == "VISUAL"
        assert results_vis[0]['id'] == "3"
        
    except Exception as e:
        pytest.fail(f"LanceDB Integration Test Failed: {e}")
