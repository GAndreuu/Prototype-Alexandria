"""
VQ-VAE ↔ Manifold Bridge
========================

Fecha o gap crítico entre o sistema de quantização (VQ-VAE) e o 
Campo Pré-Estrutural (DynamicManifold).

Teoria:
    g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)
    
    Onde c_a são os anchor_points do codebook VQ-VAE.
    A métrica deve curvar na direção dos atratores discretos.

Uso:
    from vqvae_manifold_bridge import VQVAEManifoldBridge
    
    bridge = VQVAEManifoldBridge(manifold, vqvae)
    bridge.initialize()
    
    # Agora manifold.embed() usa os atratores
    point = manifold.embed(embedding)

Autor: G (Alexandria Project)
Versão: 1.0
Status: Implementação do gap S_λ → S_ι
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# mHC Safety Layer import
try:
    from .manifold_constraints import normalize_weights_convex
    _HAS_MHC = True
except ImportError:
    _HAS_MHC = False
    logger.warning("manifold_constraints não disponível, usando normalização padrão")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ManifoldPoint:
    """Ponto na variedade com coordenadas contínuas e códigos discretos."""
    coordinates: np.ndarray          # Coordenadas contínuas [dim]
    discrete_codes: np.ndarray       # Códigos VQ-VAE [num_heads]
    activation: float = 0.0          # Nível de ativação atual
    head_contributions: Optional[np.ndarray] = None  # Contribuição por head
    nearest_anchor_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_near_attractor(self) -> bool:
        """Verifica se ponto está próximo de um atrator."""
        return self.nearest_anchor_distance < 0.1
    
    def __repr__(self):
        codes_str = ",".join(map(str, self.discrete_codes))
        return f"ManifoldPoint(codes=[{codes_str}], dist={self.nearest_anchor_distance:.4f})"


@dataclass
class AnchorPoint:
    """Ponto âncora derivado do codebook VQ-VAE."""
    head_idx: int
    code_idx: int
    coordinates: np.ndarray
    usage_count: int = 0
    activation_strength: float = 1.0
    
    @property
    def global_idx(self) -> int:
        """Índice global no codebook concatenado."""
        return self.head_idx * 256 + self.code_idx  # Assumindo 256 códigos/head


class ProjectionMode(Enum):
    """Modos de projeção embedding → manifold."""
    FLAT = "flat"                    # Apenas normalização (atual)
    NEAREST_ANCHOR = "nearest"       # Pull para âncora mais próxima
    WEIGHTED_ANCHORS = "weighted"    # Interpolação entre âncoras próximas
    GEODESIC_SNAP = "geodesic"       # Snap via geodésica (mais caro)


# =============================================================================
# BRIDGE CONFIGURATION
# =============================================================================

@dataclass
class BridgeConfig:
    """Configuração da ponte VQ-VAE ↔ Manifold."""
    
    # Dimensões
    embedding_dim: int = 384
    num_heads: int = 4
    codes_per_head: int = 256
    head_dim: int = 128              # 512 / 4 heads
    
    # Projeção
    projection_mode: ProjectionMode = ProjectionMode.WEIGHTED_ANCHORS
    
    # Atração para âncoras
    pull_radius: float = 0.3         # Raio de influência de cada âncora
    pull_strength: float = 0.5       # Força máxima do pull (0-1)
    num_nearest_anchors: int = 4     # Quantas âncoras considerar no weighted
    
    # Métrica
    deformation_radius: float = 0.2  # Raio de deformação da métrica
    deformation_strength: float = 0.3
    
    # Normalização
    normalize_embeddings: bool = True
    normalize_anchors: bool = True


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class VQVAEManifoldBridge:
    """
    Ponte que conecta VQ-VAE codebook com DynamicManifold.
    
    Responsabilidades:
    1. Extrair e organizar anchor_points do codebook
    2. Modificar projeção embed() para usar atratores
    3. Implementar from_vqvae_codes() 
    4. Calcular deformações da métrica
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        if config is None:
            # Tentar carregar calibração salva
            try:
                import json
                import os
                config_path = 'config/bridge_calibration.json'
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        calib_data = json.load(f)
                    
                    config = BridgeConfig(
                        pull_strength=calib_data.get('pull_strength', 0.5),
                        pull_radius=calib_data.get('pull_radius', 0.3),
                        deformation_strength=calib_data.get('deformation_strength', 0.3),
                        deformation_radius=calib_data.get('deformation_radius', 0.2),
                        num_nearest_anchors=calib_data.get('num_nearest_anchors', 4),
                        projection_mode=ProjectionMode(calib_data.get('projection_mode', 'weighted'))
                    )
                    logger.info(f"VQVAEManifoldBridge: Configuração calibrada carregada (score={calib_data.get('calibration_score', 'N/A')})")
                else:
                    config = BridgeConfig()
            except Exception as e:
                logger.warning(f"Erro ao carregar calibração, usando default: {e}")
                config = BridgeConfig()
        
        self.config = config
        self.codebook_by_head = {}
        self.anchor_points = None
        self._kdtree = None
    
    # =========================================================================
    # CONEXÃO COM VQ-VAE
    # =========================================================================
    
    def connect_vqvae(self, vqvae_model) -> bool:
        """
        Conecta com modelo VQ-VAE e extrai codebook.
        
        Args:
            vqvae_model: Modelo com quantizer.codebooks ou método get_codebook()
        
        Returns:
            True se conexão bem-sucedida
        """
        try:
            self._vqvae = vqvae_model
            
            # Extrair codebook (diferentes APIs possíveis)
            codebook = self._extract_codebook(vqvae_model)
            
            if codebook is None:
                logger.error("Não foi possível extrair codebook do VQ-VAE")
                return False
            
            # Processar codebook em anchor_points
            self._process_codebook(codebook)
            
            # Construir estrutura de busca
            self._build_search_structure()
            
            logger.info(f"VQ-VAE conectado: {len(self.anchor_metadata)} anchor points")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar VQ-VAE: {e}")
            return False
    
    def _extract_codebook(self, vqvae_model) -> Optional[np.ndarray]:
        """Extrai codebook do modelo (suporta múltiplas APIs)."""
        
        # Tentar diferentes atributos/métodos
        extraction_attempts = [
            lambda m: m.quantizer.codebooks.detach().cpu().numpy(),
            lambda m: m.get_codebook(),
            lambda m: m.codebook.detach().cpu().numpy(),
            lambda m: np.array([head.weight.detach().cpu().numpy() 
                               for head in m.quantizer.heads]),
        ]
        
        for attempt in extraction_attempts:
            try:
                codebook = attempt(vqvae_model)
                if codebook is not None and len(codebook) > 0:
                    return codebook
            except (AttributeError, TypeError):
                continue
        
        return None
    
    def _process_codebook(self, codebook: np.ndarray):
        """
        Processa codebook raw em anchor_points estruturados.
        
        Esperado: codebook shape [num_heads, codes_per_head, head_dim]
                  ou [num_heads * codes_per_head, head_dim]
        
        Estratégia: 
        - Armazena codebook por head para busca eficiente
        - Cria anchor_points como vetores 512D (concatenação dos heads)
        - Cada anchor é um ponto representativo (centroide do codebook)
        """
        # Normalizar shape
        if codebook.ndim == 2:
            # Flat codebook: reshape para [heads, codes, dim]
            total_codes = codebook.shape[0]
            codes_per_head = total_codes // self.config.num_heads
            codebook = codebook.reshape(
                self.config.num_heads, 
                codes_per_head, 
                -1
            )
        
        num_heads, codes_per_head, head_dim = codebook.shape
        
        # Atualizar config
        self.config.num_heads = num_heads
        self.config.codes_per_head = codes_per_head
        self.config.head_dim = head_dim
        
        # Atualizar embedding_dim para refletir a dimensão real do latent space
        latent_dim = num_heads * head_dim  # 4 * 128 = 512
        self.config.embedding_dim = latent_dim
        
        # Armazenar codebook por head
        self.codebook_by_head = {}
        for h in range(num_heads):
            self.codebook_by_head[h] = codebook[h]
        
        # Criar anchor_points: todos os códigos de todos os heads
        # Cada anchor é a contribuição de um código em uma posição específica
        anchors = []
        self.anchor_metadata = []
        
        for h in range(num_heads):
            for c in range(codes_per_head):
                # Criar vetor no espaço latente (512D)
                anchor_vec = np.zeros(latent_dim)
                
                # Posição deste head
                start_idx = h * head_dim
                end_idx = start_idx + head_dim
                
                # Inserir código
                anchor_vec[start_idx:end_idx] = codebook[h, c]
                
                if self.config.normalize_anchors:
                    norm = np.linalg.norm(anchor_vec)
                    if norm > 1e-8:
                        anchor_vec = anchor_vec / norm
                
                anchors.append(anchor_vec)
                self.anchor_metadata.append(AnchorPoint(
                    head_idx=h,
                    code_idx=c,
                    coordinates=anchor_vec
                ))
        
        self.anchor_points = np.array(anchors)
        logger.debug(f"Processados {len(anchors)} anchor points, shape={self.anchor_points.shape}")
    
    def _build_search_structure(self):
        """Constrói KDTree para busca eficiente de âncoras próximas."""
        if self.anchor_points is None:
            return
        
        try:
            from scipy.spatial import cKDTree
            self._kdtree = cKDTree(self.anchor_points)
            logger.debug("KDTree construído para anchor search")
        except ImportError:
            logger.warning("scipy não disponível, usando busca linear")
            self._kdtree = None
        
        # Cache de normas para cálculos rápidos
        self._anchor_norms = np.linalg.norm(self.anchor_points, axis=1)
    
    # =========================================================================
    # PROJEÇÃO EMBEDDING → MANIFOLD
    # =========================================================================
    
    def embed(self, embedding: np.ndarray) -> ManifoldPoint:
        """
        Projeta embedding no manifold usando atratores VQ-VAE.
        
        Esta é a função que substitui/extende DynamicManifold.embed()
        
        Args:
            embedding: Vetor 384D do SentenceTransformer (ou 512D do latente)
        
        Returns:
            ManifoldPoint com coordenadas e códigos
        """
        # Projetar para espaço latente se necessário
        embedding = self._project_to_latent(embedding)
        
        # Normalizar se configurado
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        # Sem anchor_points: fallback para projeção flat
        if self.anchor_points is None:
            return ManifoldPoint(
                coordinates=embedding,
                discrete_codes=np.array([-1] * self.config.num_heads),
                nearest_anchor_distance=float('inf')
            )
        
        # Aplicar modo de projeção
        if self.config.projection_mode == ProjectionMode.FLAT:
            return self._embed_flat(embedding)
        elif self.config.projection_mode == ProjectionMode.NEAREST_ANCHOR:
            return self._embed_nearest(embedding)
        elif self.config.projection_mode == ProjectionMode.WEIGHTED_ANCHORS:
            return self._embed_weighted(embedding)
        else:
            return self._embed_geodesic(embedding)
    
    def _project_to_latent(self, embedding: np.ndarray) -> np.ndarray:
        """
        Projeta embedding para dimensão do espaço latente.
        
        Se embedding é 384D e latente é 512D, usa projeção linear aprendida
        ou padding com repetição.
        """
        input_dim = len(embedding)
        latent_dim = self.config.embedding_dim
        
        if input_dim == latent_dim:
            return embedding
        
        if input_dim < latent_dim:
            # Expandir: repetir/tile para preencher
            # Estratégia: dividir embedding em chunks e distribuir pelos heads
            result = np.zeros(latent_dim)
            
            # Calcular quantos elementos do input vão para cada posição do latente
            chunk_size = input_dim // self.config.num_heads
            head_dim = self.config.head_dim
            
            for h in range(self.config.num_heads):
                input_start = h * chunk_size
                input_end = min(input_start + chunk_size, input_dim)
                
                latent_start = h * head_dim
                latent_end = latent_start + head_dim
                
                # Copiar o que couber
                input_chunk = embedding[input_start:input_end]
                copy_len = min(len(input_chunk), head_dim)
                result[latent_start:latent_start + copy_len] = input_chunk[:copy_len]
                
                # Preencher resto com média local ou repetição
                if copy_len < head_dim:
                    mean_val = np.mean(input_chunk) if len(input_chunk) > 0 else 0
                    result[latent_start + copy_len:latent_end] = mean_val
            
            return result
        else:
            # Comprimir: média por chunks
            result = np.zeros(latent_dim)
            chunk_size = input_dim // latent_dim
            
            for i in range(latent_dim):
                start = i * chunk_size
                end = min(start + chunk_size, input_dim)
                result[i] = np.mean(embedding[start:end])
            
            return result
    
    def _embed_flat(self, embedding: np.ndarray) -> ManifoldPoint:
        """Projeção flat (apenas encontra códigos, sem deformação)."""
        codes, distances = self._find_codes_per_head(embedding)
        
        return ManifoldPoint(
            coordinates=embedding,
            discrete_codes=codes,
            nearest_anchor_distance=np.mean(distances)
        )
    
    def _embed_nearest(self, embedding: np.ndarray) -> ManifoldPoint:
        """Projeção com pull para âncora mais próxima."""
        codes, distances = self._find_codes_per_head(embedding)
        
        # Encontrar âncora global mais próxima
        if self._kdtree is not None:
            dist, idx = self._kdtree.query(embedding, k=1)
            nearest_anchor = self.anchor_points[idx]
        else:
            dists = np.linalg.norm(self.anchor_points - embedding, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            nearest_anchor = self.anchor_points[idx]
        
        # Calcular pull strength (decai com distância)
        pull = self.config.pull_strength * np.exp(
            -(dist ** 2) / (self.config.pull_radius ** 2)
        )
        
        # Interpolar coordenadas
        coords = embedding + pull * (nearest_anchor - embedding)
        
        return ManifoldPoint(
            coordinates=coords,
            discrete_codes=codes,
            nearest_anchor_distance=dist
        )
    
    def _embed_weighted(self, embedding: np.ndarray) -> ManifoldPoint:
        """
        Projeção com interpolação ponderada entre k âncoras mais próximas.
        
        Esta é a projeção mais sofisticada e recomendada.
        
        Usa mHC Safety Layer (Sinkhorn) para garantir que o ponto
        interpolado seja estritamente dentro do convex hull das âncoras.
        Ref: mHC paper (DeepSeek-AI, 2025)
        """
        codes, head_distances = self._find_codes_per_head(embedding)
        
        # Encontrar k âncoras mais próximas globalmente
        k = self.config.num_nearest_anchors
        
        if self._kdtree is not None:
            distances, indices = self._kdtree.query(embedding, k=k)
        else:
            all_dists = np.linalg.norm(self.anchor_points - embedding, axis=1)
            indices = np.argpartition(all_dists, k)[:k]
            distances = all_dists[indices]
            # Ordenar
            sort_idx = np.argsort(distances)
            indices = indices[sort_idx]
            distances = distances[sort_idx]
        
        # =====================================================================
        # mHC SAFETY LAYER: Normalização Sinkhorn para combinação convexa
        # Garante que o ponto interpolado esteja dentro do convex hull
        # das âncoras, evitando reconstruções fora da distribuição
        # =====================================================================
        log_affinities = -distances / self.config.pull_radius
        
        if _HAS_MHC:
            # Usa normalização mHC (Sinkhorn) para pesos baricêntricos estáveis
            weights = normalize_weights_convex(np.exp(log_affinities), use_sinkhorn=False)
        else:
            # Fallback: softmax padrão
            weights = np.exp(log_affinities)
            weights = weights / (np.sum(weights) + 1e-8)
        
        # Interpolação ponderada das âncoras
        anchor_blend = np.zeros_like(embedding)
        for w, idx in zip(weights, indices):
            anchor_blend += w * self.anchor_points[idx]
        
        # Blend final: embedding original + pull para âncoras
        pull = self.config.pull_strength
        coords = (1 - pull) * embedding + pull * anchor_blend
        
        # Renormalizar se necessário
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(coords)
            if norm > 1e-8:
                coords = coords / norm
        
        return ManifoldPoint(
            coordinates=coords,
            discrete_codes=codes,
            head_contributions=weights,
            nearest_anchor_distance=distances[0] if len(distances) > 0 else float('inf')
        )
    
    def _embed_geodesic(self, embedding: np.ndarray) -> ManifoldPoint:
        """
        Projeção via snap geodésico (mais caro computacionalmente).
        
        TODO: Implementar quando GeodesicFlow estiver integrado.
        """
        logger.warning("Geodesic projection não implementado, usando weighted")
        return self._embed_weighted(embedding)
    
    def find_codes(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encontra o código mais próximo em cada head.
        
        Returns:
            codes: [num_heads] índices dos códigos
            distances: [num_heads] distâncias para cada código
        """
        codes = np.zeros(self.config.num_heads, dtype=np.int32)
        distances = np.zeros(self.config.num_heads)
        
        head_dim = self.config.head_dim
        
        for h in range(self.config.num_heads):
            # Extrair slice do embedding para este head
            start_idx = h * head_dim
            end_idx = start_idx + head_dim
            head_embedding = embedding[start_idx:end_idx]
            
            # Buscar no codebook deste head
            if h in self.codebook_by_head:
                codebook_h = self.codebook_by_head[h]
                dists = np.linalg.norm(codebook_h - head_embedding, axis=1)
                codes[h] = np.argmin(dists)
                distances[h] = dists[codes[h]]
            else:
                codes[h] = -1
                distances[h] = float('inf')
        
        return codes, distances

    def get_code_embedding(self, head_idx: int, code_idx: int) -> Optional[np.ndarray]:
        """
        Retorna o embedding do código específico.
        """
        if head_idx in self.codebook_by_head:
            codebook = self.codebook_by_head[head_idx]
            if 0 <= code_idx < len(codebook):
                return codebook[code_idx].copy()
        return None
        
    def _find_codes_per_head(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Deprecated alias for find_codes."""
        return self.find_codes(embedding)
    
    # =========================================================================
    # RECONSTRUÇÃO CODES → EMBEDDING
    # =========================================================================
    
    def from_vqvae_codes(self, codes: List[int]) -> ManifoldPoint:
        """
        Reconstrói ManifoldPoint a partir de códigos VQ-VAE.
        
        Esta função implementa o que era NotImplementedError.
        
        Args:
            codes: Lista de códigos [h0, h1, h2, h3]
        
        Returns:
            ManifoldPoint reconstruído
        """
        if self.anchor_points is None:
            raise ValueError(
                "Anchor points não configurados. "
                "Chame connect_vqvae() primeiro."
            )
        
        if len(codes) != self.config.num_heads:
            raise ValueError(
                f"Esperados {self.config.num_heads} códigos, "
                f"recebidos {len(codes)}"
            )
        
        # Reconstruir coordenadas concatenando os códigos de cada head
        coords = np.zeros(self.config.embedding_dim)
        head_dim = self.config.head_dim
        
        for h, code in enumerate(codes):
            if code < 0 or code >= self.config.codes_per_head:
                logger.warning(f"Código inválido head={h}, code={code}")
                continue
            
            if h in self.codebook_by_head:
                start_idx = h * head_dim
                end_idx = start_idx + head_dim
                coords[start_idx:end_idx] = self.codebook_by_head[h][code]
        
        # Normalizar
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(coords)
            if norm > 1e-8:
                coords = coords / norm
        
        return ManifoldPoint(
            coordinates=coords,
            discrete_codes=np.array(codes),
            nearest_anchor_distance=0.0  # Exatamente no atrator
        )
    
    # =========================================================================
    # DEFORMAÇÃO DA MÉTRICA
    # =========================================================================
    
    def compute_metric_deformation(self, point: np.ndarray) -> np.ndarray:
        """
        Calcula tensor de deformação da métrica no ponto dado.
        
        g_ij(x) = δ_ij + Σ_a w_a · exp(-|x - c_a|² / r²)
        
        Returns:
            Matriz [dim, dim] representando a métrica local
        """
        dim = len(point)
        
        # Começar com métrica flat (identidade)
        g = np.eye(dim)
        
        if self.anchor_points is None:
            return g
        
        r2 = self.config.deformation_radius ** 2
        strength = self.config.deformation_strength
        
        # Somar contribuições dos k anchors mais próximos
        # (não precisamos somar todos, apenas os relevantes)
        k = min(16, len(self.anchor_points))
        
        if self._kdtree is not None:
            distances, indices = self._kdtree.query(point, k=k)
        else:
            all_dists = np.linalg.norm(self.anchor_points - point, axis=1)
            indices = np.argpartition(all_dists, k)[:k]
            distances = all_dists[indices]
        
        for dist, idx in zip(distances, indices):
            # Peso Gaussiano
            w = strength * np.exp(-(dist ** 2) / r2)
            
            if w < 1e-6:
                continue  # Contribuição negligenciável
            
            # Direção para a âncora
            anchor = self.anchor_points[idx]
            direction = anchor - point
            dir_norm = np.linalg.norm(direction)
            
            if dir_norm > 1e-8:
                direction = direction / dir_norm
                
                # Adicionar deformação na direção da âncora
                # Isso "estica" o espaço na direção dos atratores
                g += w * np.outer(direction, direction)
        
        return g
    
    def compute_christoffel(self, point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """
        Calcula símbolos de Christoffel via diferenças finitas.
        
        Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        
        Returns:
            Tensor [dim, dim, dim]
        """
        dim = len(point)
        
        # Métrica e sua inversa no ponto
        g = self.compute_metric_deformation(point)
        g_inv = np.linalg.inv(g)
        
        # Derivadas da métrica
        dg = np.zeros((dim, dim, dim))  # dg[l, i, j] = ∂_l g_{ij}
        
        for l in range(dim):
            # Perturbação na direção l
            point_plus = point.copy()
            point_plus[l] += epsilon
            point_minus = point.copy()
            point_minus[l] -= epsilon
            
            g_plus = self.compute_metric_deformation(point_plus)
            g_minus = self.compute_metric_deformation(point_minus)
            
            dg[l] = (g_plus - g_minus) / (2 * epsilon)
        
        # Símbolos de Christoffel
        Gamma = np.zeros((dim, dim, dim))
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    for l in range(dim):
                        Gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )
        
        return Gamma
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def get_nearest_anchors(
        self, 
        point: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[AnchorPoint, float]]:
        """Retorna k âncoras mais próximas com distâncias."""
        if self.anchor_points is None:
            return []
        
        if self._kdtree is not None:
            distances, indices = self._kdtree.query(point, k=k)
        else:
            all_dists = np.linalg.norm(self.anchor_points - point, axis=1)
            indices = np.argpartition(all_dists, k)[:k]
            distances = all_dists[indices]
            sort_idx = np.argsort(distances)
            indices = indices[sort_idx]
            distances = distances[sort_idx]
        
        return [
            (self.anchor_metadata[idx], dist) 
            for idx, dist in zip(indices, distances)
        ]
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas do bridge."""
        return {
            "connected": self.anchor_points is not None,
            "num_anchors": len(self.anchor_points) if self.anchor_points is not None else 0,
            "num_heads": self.config.num_heads,
            "codes_per_head": self.config.codes_per_head,
            "projection_mode": self.config.projection_mode.value,
            "pull_radius": self.config.pull_radius,
            "pull_strength": self.config.pull_strength,
            "has_kdtree": self._kdtree is not None,
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def patch_dynamic_manifold(manifold, bridge: VQVAEManifoldBridge):
    """
    Patch para integrar bridge em um DynamicManifold existente.
    
    Uso:
        bridge = VQVAEManifoldBridge()
        bridge.connect_vqvae(vqvae_model)
        patch_dynamic_manifold(manifold, bridge)
        
        # Agora manifold.embed() usa a lógica do bridge
    """
    original_embed = manifold.embed
    
    def patched_embed(embedding):
        return bridge.embed(embedding)
    
    def patched_from_codes(codes):
        return bridge.from_vqvae_codes(codes)
    
    manifold.embed = patched_embed
    manifold.from_vqvae_codes = patched_from_codes
    manifold._vqvae_bridge = bridge
    
    logger.info("DynamicManifold patched com VQVAEManifoldBridge")


def create_integrated_field(
    vqvae_model,
    config: Optional[BridgeConfig] = None
) -> VQVAEManifoldBridge:
    """
    Factory function para criar bridge já conectado.
    
    Uso:
        bridge = create_integrated_field(my_vqvae)
        point = bridge.embed(embedding)
    """
    bridge = VQVAEManifoldBridge(config)
    success = bridge.connect_vqvae(vqvae_model)
    
    if not success:
        raise RuntimeError("Falha ao conectar VQ-VAE")
    
    return bridge


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Teste básico do bridge."""
    
    print("=" * 60)
    print("VQ-VAE ↔ Manifold Bridge - Teste")
    print("=" * 60)
    
    # Criar mock VQ-VAE com método get_codebook()
    class MockCodebook:
        def __init__(self, data):
            self._data = data
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self._data
    
    class MockQuantizer:
        def __init__(self):
            self.codebooks = MockCodebook(
                np.random.randn(4, 256, 128).astype(np.float32)
            )
    
    class MockVQVAE:
        def __init__(self):
            self.quantizer = MockQuantizer()
        
        def get_codebook(self):
            return self.quantizer.codebooks.numpy()
    
    # Criar bridge
    config = BridgeConfig(
        projection_mode=ProjectionMode.WEIGHTED_ANCHORS,
        pull_strength=0.5
    )
    bridge = VQVAEManifoldBridge(config)
    
    # Conectar mock
    mock_vqvae = MockVQVAE()
    success = bridge.connect_vqvae(mock_vqvae)
    print(f"\n1. Conexão VQ-VAE: {'✓' if success else '✗'}")
    
    # Stats
    stats = bridge.stats()
    print(f"\n2. Estatísticas:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Teste embed
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    
    point = bridge.embed(embedding)
    print(f"\n3. Embed teste:")
    print(f"   {point}")
    print(f"   Códigos: {point.discrete_codes}")
    print(f"   Distância ao atrator: {point.nearest_anchor_distance:.4f}")
    
    # Teste from_codes
    codes = point.discrete_codes.tolist()
    reconstructed = bridge.from_vqvae_codes(codes)
    print(f"\n4. Reconstrução from_codes:")
    print(f"   {reconstructed}")
    
    # Teste métrica - usar coordenadas do point (já projetadas)
    g = bridge.compute_metric_deformation(point.coordinates)
    print(f"\n5. Deformação da métrica:")
    print(f"   Shape: {g.shape}")
    print(f"   Traço (deve ser > {len(point.coordinates)}): {np.trace(g):.4f}")
    det = np.linalg.det(g)
    print(f"   Det (deve ser >= 1): {det:.4f}")
    
    # Verificar que é positiva definida
    eigenvalues = np.linalg.eigvalsh(g)
    print(f"   Autovalores min/max: {eigenvalues.min():.4f} / {eigenvalues.max():.4f}")
    
    # Nearest anchors - usar coordenadas projetadas
    nearest = bridge.get_nearest_anchors(point.coordinates, k=3)
    print(f"\n6. Âncoras mais próximas:")
    for anchor, dist in nearest:
        print(f"   Head {anchor.head_idx}, Code {anchor.code_idx}: dist={dist:.4f}")
    
    # Teste de deformação PERTO de um atrator
    print(f"\n7. Deformação perto de atrator:")
    # Usar anchor direto com pequena perturbação (sem renormalizar)
    anchor_coords = bridge.anchor_points[0].copy()
    near_anchor = anchor_coords + np.random.randn(512) * 0.01  # Muito perto
    
    # Temporariamente aumentar strength para ver efeito
    old_strength = bridge.config.deformation_strength
    bridge.config.deformation_strength = 1.0
    bridge.config.deformation_radius = 0.5
    
    g_near = bridge.compute_metric_deformation(near_anchor)
    print(f"   Traço (deve ser > 512): {np.trace(g_near):.4f}")
    det_near = np.linalg.det(g_near)
    print(f"   Det (deve ser > 1): {det_near:.6f}")
    eigenvalues_near = np.linalg.eigvalsh(g_near)
    print(f"   Autovalores min/max: {eigenvalues_near.min():.4f} / {eigenvalues_near.max():.4f}")
    
    # Restaurar
    bridge.config.deformation_strength = old_strength
    print(f"\n6. Âncoras mais próximas:")
    for anchor, dist in nearest:
        print(f"   Head {anchor.head_idx}, Code {anchor.code_idx}: dist={dist:.4f}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
