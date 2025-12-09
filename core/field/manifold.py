"""
DynamicManifold: Variedade Diferenciável com Dimensão Variável
==============================================================

A variedade é o espaço onde o conhecimento vive. Não é um grafo discreto,
é um espaço contínuo que pode expandir e contrair.

Conceitos-chave:
- Pontos na variedade = conceitos/embeddings
- Códigos VQ-VAE = coordenadas discretas (âncoras)
- Dimensão pode crescer durante expansão
- Topologia emerge da distribuição de pontos

Conexão com VQ-VAE:
    O VQ-VAE comprime 384D → 4 códigos (4 heads × 256 valores)
    Esses códigos são coordenadas discretas na variedade
    A variedade interpola entre essas âncoras
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, lil_matrix


@dataclass
class ManifoldConfig:
    """Configuração da variedade."""
    base_dim: int = 384              # Dimensão base (embedding size)
    num_heads: int = 4               # Heads do VQ-VAE
    codebook_size: int = 256         # Códigos por head
    max_expansion: int = 128         # Máximo de dimensões extras
    sparsity_threshold: float = 0.01 # Threshold para considerar ponto ativo
    neighborhood_k: int = 16         # K vizinhos para estrutura local


@dataclass 
class ManifoldPoint:
    """Um ponto na variedade."""
    coordinates: np.ndarray          # Coordenadas contínuas
    discrete_codes: np.ndarray       # Códigos VQ-VAE originais [4]
    activation: float = 0.0          # Nível de ativação atual
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicManifold:
    """
    Variedade diferenciável com dimensão variável.
    
    A variedade começa com dimensão base (384D dos embeddings) e pode
    expandir durante o ciclo de processamento para acomodar estruturas
    emergentes.
    
    Attributes:
        config: Configuração da variedade
        current_dim: Dimensão atual (pode mudar)
        points: Dicionário de pontos {id: ManifoldPoint}
        anchor_points: Pontos âncora derivados do codebook VQ-VAE
        kdtree: Estrutura para busca de vizinhos (rebuild on change)
    """
    
    def __init__(self, config: Optional[ManifoldConfig] = None):
        self.config = config or ManifoldConfig()
        self.current_dim = self.config.base_dim
        self.points: Dict[str, ManifoldPoint] = {}
        self.anchor_points: np.ndarray = None  # [num_anchors, dim]
        self._kdtree: Optional[KDTree] = None
        self._dirty = True  # Flag para rebuild de estruturas
        
        # Dimensões expandidas (inicialmente vazio)
        self._expansion_dims: List[np.ndarray] = []
        
    # =========================================================================
    # EMBEDDING E PROJEÇÃO
    # =========================================================================
    
    def embed(self, embedding: np.ndarray, codes: Optional[np.ndarray] = None) -> ManifoldPoint:
        """
        Projeta um embedding 384D em um ponto da variedade.
        
        Args:
            embedding: Vetor 384D (output do SentenceTransformer)
            codes: Códigos VQ-VAE opcionais [4] para ancorar o ponto
            
        Returns:
            ManifoldPoint na variedade
            
        TODO:
            - Se temos dimensões expandidas, projetar nelas também
            - Usar códigos VQ-VAE para determinar região
        """
        # Normaliza embedding
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Se há dimensões expandidas, pad com zeros (ou projeção)
        if self.current_dim > self.config.base_dim:
            expanded = np.zeros(self.current_dim)
            expanded[:self.config.base_dim] = norm_embedding
            # TODO: Projetar nas dimensões extras baseado em contexto
            coordinates = expanded
        else:
            coordinates = norm_embedding
            
        # Cria ponto
        point = ManifoldPoint(
            coordinates=coordinates,
            discrete_codes=codes if codes is not None else np.zeros(self.config.num_heads, dtype=np.int32),
            activation=0.0
        )
        
        return point
    
    def from_vqvae_codes(self, codes: np.ndarray) -> ManifoldPoint:
        """
        Cria ponto a partir apenas dos códigos VQ-VAE.
        
        Os códigos [4] definem uma região discreta no espaço.
        Útil quando não temos o embedding original.
        
        Args:
            codes: Array [4] com índices do codebook (0-255 cada)
            
        Returns:
            ManifoldPoint na região definida pelos códigos
            
        TODO:
            - Usar centroides do codebook VQ-VAE para reconstruir coordenadas
            - Interpolar se tivermos anchor_points
        """
        if self.anchor_points is None:
            # Sem âncoras, criar coordenadas sintéticas
            # Cada código contribui para uma região do espaço
            coordinates = np.zeros(self.current_dim)
            
            chunk_size = self.config.base_dim // self.config.num_heads
            for i, code in enumerate(codes):
                # Distribui código em região do espaço
                start = i * chunk_size
                end = start + chunk_size
                # TODO: Usar codebook real do VQ-VAE
                coordinates[start:end] = np.sin(code * np.linspace(0, np.pi, chunk_size))
                
            coordinates = coordinates / (np.linalg.norm(coordinates) + 1e-8)
        else:
            # Usar âncoras reais
            # TODO: Implementar lookup no codebook
            raise NotImplementedError("Anchor-based embedding pendente")
            
        return ManifoldPoint(
            coordinates=coordinates,
            discrete_codes=codes,
            activation=0.0
        )
    
    def set_anchor_points(self, codebook_vectors: np.ndarray):
        """
        Define pontos âncora a partir do codebook VQ-VAE treinado.
        
        Args:
            codebook_vectors: [num_heads, codebook_size, head_dim]
                             ou [total_codes, embedding_dim]
                             
        TODO:
            - Processar formato do codebook do MonolithVQVAE
            - Criar estrutura de lookup eficiente
        """
        self.anchor_points = codebook_vectors
        self._dirty = True
        
    # =========================================================================
    # OPERAÇÕES DE PONTO
    # =========================================================================
    
    def add_point(self, point_id: str, point: ManifoldPoint):
        """Adiciona ponto à variedade."""
        self.points[point_id] = point
        self._dirty = True
        
    def get_point(self, point_id: str) -> Optional[ManifoldPoint]:
        """Recupera ponto por ID."""
        return self.points.get(point_id)
    
    def activate_point(self, point_id: str, intensity: float = 1.0):
        """
        Ativa um ponto (trigger).
        
        Ativação é o que causa deformação na métrica.
        
        Args:
            point_id: ID do ponto
            intensity: Intensidade da ativação [0, 1+]
        """
        if point_id in self.points:
            self.points[point_id].activation = intensity
            
    def decay_activations(self, rate: float = 0.1):
        """Decai todas as ativações (relaxamento)."""
        for point in self.points.values():
            point.activation *= (1 - rate)
            if point.activation < self.config.sparsity_threshold:
                point.activation = 0.0
                
    # =========================================================================
    # ESTRUTURA E VIZINHANÇA
    # =========================================================================
    
    def _rebuild_kdtree(self):
        """Reconstrói KDTree para busca de vizinhos."""
        if len(self.points) == 0:
            self._kdtree = None
            return
            
        coords = np.array([p.coordinates for p in self.points.values()])
        self._kdtree = KDTree(coords)
        self._dirty = False
        
    def get_neighbors(self, point: ManifoldPoint, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Encontra k vizinhos mais próximos de um ponto.
        
        Args:
            point: Ponto de referência
            k: Número de vizinhos (default: config.neighborhood_k)
            
        Returns:
            Lista de (point_id, distância)
        """
        if self._dirty:
            self._rebuild_kdtree()
            
        if self._kdtree is None:
            return []
        
        if len(self.points) == 0:
            return []
            
        k = k or self.config.neighborhood_k
        k = min(k, len(self.points))
        
        if k == 0:
            return []
        
        distances, indices = self._kdtree.query(point.coordinates, k=k)
        
        # KDTree retorna escalar quando k=1, precisamos de array
        if k == 1:
            distances = [distances]
            indices = [indices]
        
        point_ids = list(self.points.keys())
        return [(point_ids[int(i)], float(d)) for i, d in zip(indices, distances)]
    
    def get_active_points(self) -> List[Tuple[str, ManifoldPoint]]:
        """Retorna pontos com ativação acima do threshold."""
        return [
            (pid, p) for pid, p in self.points.items() 
            if p.activation >= self.config.sparsity_threshold
        ]
        
    # =========================================================================
    # EXPANSÃO E CONTRAÇÃO DIMENSIONAL
    # =========================================================================
    
    def expand_dimension(self, n_dims: int = 1, basis: Optional[np.ndarray] = None):
        """
        Expande a variedade em novas dimensões.
        
        Isso acontece durante a fase de EXPANSÃO do ciclo,
        quando o sistema precisa de mais graus de liberdade.
        
        Args:
            n_dims: Número de dimensões a adicionar
            basis: Vetores base para as novas dimensões [n_dims, current_dim]
                   Se None, usa direções aleatórias ortogonais
                   
        TODO:
            - Derivar novas dimensões de estrutura emergente
            - Usar PCA dos resíduos para encontrar direções úteis
        """
        if self.current_dim + n_dims > self.config.base_dim + self.config.max_expansion:
            raise ValueError(f"Expansão excede máximo permitido")
            
        if basis is None:
            # Gera direções aleatórias ortogonais
            basis = np.random.randn(n_dims, self.current_dim)
            basis, _ = np.linalg.qr(basis.T)
            basis = basis.T[:n_dims]
            
        self._expansion_dims.append(basis)
        old_dim = self.current_dim
        self.current_dim += n_dims
        
        # Re-projeta todos os pontos nas novas dimensões
        for point in self.points.values():
            old_coords = point.coordinates
            new_coords = np.zeros(self.current_dim)
            new_coords[:old_dim] = old_coords
            # Projeção nas novas dimensões (inicialmente zero)
            # TODO: Calcular projeção baseada em contexto
            point.coordinates = new_coords
            
        self._dirty = True
        
    def contract_dimension(self, n_dims: int = 1):
        """
        Contrai a variedade removendo dimensões.
        
        Isso acontece durante a fase de COMPRESSÃO do ciclo.
        Remove dimensões com menor variância/informação.
        
        Args:
            n_dims: Número de dimensões a remover
            
        TODO:
            - Identificar dimensões com menor variância
            - Preservar informação via projeção antes de remover
        """
        if self.current_dim - n_dims < self.config.base_dim:
            raise ValueError(f"Contração abaixo da dimensão base não permitida")
            
        # Por agora, remove as últimas dimensões
        # TODO: PCA para identificar dimensões menos informativas
        self.current_dim -= n_dims
        
        for point in self.points.values():
            point.coordinates = point.coordinates[:self.current_dim]
            
        if self._expansion_dims:
            self._expansion_dims.pop()
            
        self._dirty = True
        
    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================
    
    def get_coordinates_matrix(self) -> np.ndarray:
        """Retorna matriz [n_points, dim] com todas as coordenadas."""
        if len(self.points) == 0:
            return np.zeros((0, self.current_dim))
        return np.array([p.coordinates for p in self.points.values()])
    
    def get_activations(self) -> np.ndarray:
        """Retorna vetor de ativações."""
        return np.array([p.activation for p in self.points.values()])
    
    def to_dict(self) -> Dict:
        """Serializa variedade para persistência."""
        return {
            'config': self.config.__dict__,
            'current_dim': self.current_dim,
            'points': {
                pid: {
                    'coordinates': p.coordinates.tolist(),
                    'discrete_codes': p.discrete_codes.tolist(),
                    'activation': p.activation,
                    'metadata': p.metadata
                }
                for pid, p in self.points.items()
            },
            'expansion_dims': [e.tolist() for e in self._expansion_dims]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DynamicManifold':
        """Reconstrói variedade de dados serializados."""
        config = ManifoldConfig(**data['config'])
        manifold = cls(config)
        manifold.current_dim = data['current_dim']
        
        for pid, pdata in data['points'].items():
            point = ManifoldPoint(
                coordinates=np.array(pdata['coordinates']),
                discrete_codes=np.array(pdata['discrete_codes']),
                activation=pdata['activation'],
                metadata=pdata['metadata']
            )
            manifold.points[pid] = point
            
        manifold._expansion_dims = [np.array(e) for e in data['expansion_dims']]
        manifold._dirty = True
        
        return manifold
    
    # =========================================================================
    # MÉTRICAS E DEBUG
    # =========================================================================
    
    def stats(self) -> Dict:
        """Estatísticas da variedade."""
        activations = self.get_activations()
        return {
            'num_points': len(self.points),
            'current_dim': self.current_dim,
            'base_dim': self.config.base_dim,
            'expansion_dims': self.current_dim - self.config.base_dim,
            'active_points': np.sum(activations > self.config.sparsity_threshold),
            'mean_activation': np.mean(activations) if len(activations) > 0 else 0,
            'max_activation': np.max(activations) if len(activations) > 0 else 0,
        }
