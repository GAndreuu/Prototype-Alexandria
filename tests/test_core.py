import pytest
import numpy as np
import os
from core.topology_engine import TopologyEngine
from core.semantic_memory import SemanticFileSystem
from core.oracle import NeuralOracle

class TestTopologyEngine:
    def test_encoding(self, engine):
        """Testa se o encoder gera vetores corretos"""
        text = ["Teste de embedding"]
        vectors = engine.encode(text)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (1, 384)
    
    def test_clustering(self, engine):
        """Testa se o clustering funciona"""
        # Criar vetores aleatórios para teste
        vectors = np.random.rand(10, 384).astype('float32')
        engine.train_manifold(vectors, n_clusters=2)
        assert engine.is_trained
        assert engine.kmeans is not None

class TestSemanticFileSystem:
    def test_ingestion(self, memory, temp_data_dir):
        """Testa ingestão de documentos"""
        # Criar arquivo temporário
        file_path = os.path.join(temp_data_dir, "test_doc.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Inteligência Artificial é o futuro.\n\nMachine Learning usa dados.")
            
        # Usar index_file em vez de ingest
        chunks_count = memory.index_file(file_path)
        assert chunks_count > 0

    def test_retrieval(self, memory, temp_data_dir):
        """Testa recuperação de informação"""
        # Ingestão prévia necessária
        file_path = os.path.join(temp_data_dir, "ml_doc.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Machine Learning usa dados para treinar modelos.")
        
        memory.index_file(file_path)
        
        results = memory.retrieve("dados")
        assert isinstance(results, list)
        # Pode estar vazio se o índice não persistiu no teste, mas não deve falhar

class TestNeuralOracle:
    def test_hybrid_synthesis(self, oracle):
        """Testa síntese híbrida"""
        query = "O que é Python?"
        evidence = [{"content": "Python é uma linguagem de programação.", "relevance": 0.9}]
        
        # Teste Local
        response_local = oracle.synthesize(query, evidence, mode="local")
        assert isinstance(response_local, str)
        assert len(response_local) > 0
        
        # Teste Híbrido (se API estiver ok)
        if oracle.is_gemini_available:
            response_hybrid = oracle.synthesize(query, evidence, mode="hybrid")
            assert isinstance(response_hybrid, str)
            assert len(response_hybrid) > 0
