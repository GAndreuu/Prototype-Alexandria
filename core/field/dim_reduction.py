import numpy as np
import pickle
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DimensionalityReducer:
    """
    Redutor de dimensionalidade para otimizar cálculos geodésicos.
    Transforma embeddings (384d) -> Manifold Space (32d).
    
    Usa PCA incremental para aprender com o fluxo de dados se não houver
    modelo pré-treinado.
    """
    
    def __init__(self, input_dim: int = 384, target_dim: int = 32, save_path: str = "data/pca_32d.pkl"):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.save_path = save_path
        self.pca = None
        self.is_fitted = False
        
        # Tenta carregar modelo existente
        self._load()
        
        # Fallback se não existir: Matriz aleatória ortogonal (Random Projection)
        # para garantir funcionamento imediato
        if not self.is_fitted:
            logger.info("PCA não treinado. Inicializando com Random Projection temporária.")
            rng = np.random.default_rng(42)
            # Matriz de projeção semi-ortogonal
            self.projection_matrix = rng.standard_normal((input_dim, target_dim))
            q, _ = np.linalg.qr(self.projection_matrix)
            self.projection_matrix = q[:, :target_dim]
    
    def _load(self):
        """Carrega modelo PCA salvo."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    self.pca = pickle.load(f)
                self.is_fitted = True
                logger.info("Modelo PCA 32d carregado com sucesso.")
            except Exception as e:
                logger.warning(f"Falha ao carregar PCA: {e}")
    
    def transform(self, embedding: np.ndarray) -> np.ndarray:
        """
        Projeta embedding(s) 384d -> 32d.
        
        Args:
            embedding: [batch, 384] ou [384]
            
        Returns:
            [batch, 32] ou [32]
        """
        # Garante array numpy
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        is_single = embedding.ndim == 1
        if is_single:
            embedding = embedding.reshape(1, -1)
            
        if self.is_fitted and self.pca is not None:
            reduced = self.pca.transform(embedding)
        else:
            # Fallback para Random Projection
            reduced = np.dot(embedding, self.projection_matrix)
            
        if is_single:
            return reduced.flatten()
        return reduced
        
    def fit_partial(self, embeddings: np.ndarray):
        """
        Treino incremental (se usarmos IncrementalPCA no futuro).
        Por enquanto, não implementado para manter simples.
        """
        pass
