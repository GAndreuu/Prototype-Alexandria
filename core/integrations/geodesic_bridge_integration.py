"""
Geodesic ↔ Bridge Integration
==============================

Conecta o motor de fluxo geodésico (GeodesicFlow) ao manifold curvo (Bridge).

Esta integração permite:
- Computar geodésicas no espaço semântico definido pelo VQ-VAE
- Propagar ativação entre conceitos seguindo a geometria natural
- Encontrar caminhos semânticos ótimos entre ideias
- Visualizar a estrutura do espaço latente

Exemplo de uso:
    bridge = VQVAEManifoldBridge(config)
    bridge.connect_vqvae(vqvae_model)
    
    gbi = GeodesicBridgeIntegration(bridge)
    
    # Encontrar caminho semântico entre dois conceitos
    path = gbi.semantic_path(concept_a_embedding, concept_b_embedding)
    
    # Propagar ativação de um conceito para conceitos relacionados
    activations = gbi.propagate_concept(source_embedding, radius=3)

Autor: Alexandria System
Versão: 1.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class GeodesicBridgeConfig:
    """Configuração da integração Geodesic ↔ Bridge."""
    
    # Parâmetros do fluxo geodésico
    max_geodesic_steps: int = 100
    geodesic_step_size: float = 0.01
    geodesic_tolerance: float = 1e-6
    
    # Parâmetros de propagação
    propagation_decay: float = 0.5
    propagation_steps: int = 3
    min_activation_threshold: float = 0.1
    
    # Parâmetros de busca de caminho
    path_search_iterations: int = 10
    path_smoothing: bool = True
    path_smoothing_factor: float = 0.3
    
    # Integração com atratores
    use_attractor_guidance: bool = True
    attractor_pull_strength: float = 0.2
    
    # Visualização e debug
    store_intermediate_paths: bool = False
    verbose: bool = False


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class SemanticPath:
    """Um caminho semântico entre dois conceitos."""
    start: np.ndarray
    end: np.ndarray
    points: np.ndarray  # [n_steps, dim]
    geodesic_length: float
    euclidean_length: float
    curvature_integral: float
    attractors_visited: List[int]
    converged: bool
    
    @property
    def geodesic_ratio(self) -> float:
        """Razão geodésica/euclidiana - quanto maior, mais curvo o caminho."""
        if self.euclidean_length > 1e-8:
            return self.geodesic_length / self.euclidean_length
        return 1.0
    
    @property
    def n_steps(self) -> int:
        return len(self.points)
    
    def interpolate(self, t: float) -> np.ndarray:
        """Interpola ponto no caminho (t entre 0 e 1)."""
        if len(self.points) < 2:
            return self.start
        idx = int(t * (len(self.points) - 1))
        idx = max(0, min(idx, len(self.points) - 1))
        return self.points[idx]


@dataclass
class ActivationMap:
    """Mapa de ativação propagada a partir de um conceito."""
    source: np.ndarray
    activations: Dict[int, float]  # idx -> activation_strength
    paths: List[np.ndarray]  # Caminhos percorridos (se armazenados)
    total_energy: float
    max_radius: float
    n_activated: int
    
    def get_top_activations(self, k: int = 10) -> List[Tuple[int, float]]:
        """Retorna os k conceitos mais ativados."""
        sorted_acts = sorted(self.activations.items(), key=lambda x: -x[1])
        return sorted_acts[:k]


@dataclass
class GeodesicField:
    """Campo de geodésicas emanando de um ponto."""
    center: np.ndarray
    directions: np.ndarray  # [n_directions, dim]
    paths: List[np.ndarray]
    lengths: List[float]
    curvatures: List[float]
    
    @property
    def mean_length(self) -> float:
        return np.mean(self.lengths) if self.lengths else 0.0
    
    @property
    def mean_curvature(self) -> float:
        return np.mean(self.curvatures) if self.curvatures else 0.0


# =============================================================================
# INTEGRAÇÃO PRINCIPAL
# =============================================================================

class GeodesicBridgeIntegration:
    """
    Integra GeodesicFlow com VQVAEManifoldBridge.
    
    Esta classe fornece uma interface de alto nível para:
    - Encontrar caminhos semânticos entre conceitos
    - Propagar ativação no espaço latente
    - Explorar a estrutura geodésica do manifold
    """
    
    def __init__(
        self,
        bridge,  # VQVAEManifoldBridge
        geodesic_flow=None,  # GeodesicFlow (opcional, cria automaticamente)
        config: Optional[GeodesicBridgeConfig] = None
    ):
        self.bridge = bridge
        self.config = config or GeodesicBridgeConfig()
        
        # Criar ou usar GeodesicFlow
        if geodesic_flow is not None:
            self.geodesic = geodesic_flow
        else:
            self._init_geodesic_flow()
        
        # Cache de caminhos
        self._path_cache: Dict[Tuple[int, int], SemanticPath] = {}
        
        logger.info("GeodesicBridgeIntegration initialized")
    
    def _init_geodesic_flow(self):
        """Inicializa GeodesicFlow a partir do bridge."""
        try:
            from core.field.geodesic_flow import GeodesicFlow, GeodesicConfig
            from core.field.manifold import DynamicManifold, ManifoldConfig
            from core.field.metric import RiemannianMetric
            
            # Criar manifold e métrica a partir do bridge
            dim = self.bridge.latent_dim if hasattr(self.bridge, 'latent_dim') else 128
            
            # Use ManifoldConfig with base_dim parameter
            manifold_config = ManifoldConfig(base_dim=dim)
            manifold = DynamicManifold(config=manifold_config)
            metric = RiemannianMetric(manifold)
            
            # Configurar geodésicas
            # Mapeando campos do GeodesicBridgeConfig para GeodesicConfig (novo)
            geo_config = GeodesicConfig(
                max_steps=self.config.max_geodesic_steps,
                dt=self.config.geodesic_step_size,
                tol=self.config.geodesic_tolerance,
                # Defaults robustos para shooting method
                shooting_iters=self.config.path_search_iterations, 
                active_dims=32, # Razoável para produção
                energy_renorm=True
            )
            
            self.geodesic = GeodesicFlow(manifold, metric, geo_config)
            
        except ImportError as e:
            logger.warning(f"Could not import GeodesicFlow: {e}")
            self.geodesic = None
    
    # =========================================================================
    # CAMINHOS SEMÂNTICOS
    # =========================================================================
    
    def semantic_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        use_cache: bool = True
    ) -> SemanticPath:
        """
        Encontra o caminho semântico (geodésica) entre dois conceitos.
        
        Args:
            start: Embedding do conceito inicial
            end: Embedding do conceito final
            use_cache: Se True, usa cache de caminhos
            
        Returns:
            SemanticPath com o caminho encontrado
        """
        # Projetar para espaço latente se necessário
        start_latent = self._to_latent(start)
        end_latent = self._to_latent(end)
        
        # Verificar cache
        cache_key = (hash(start_latent.tobytes()), hash(end_latent.tobytes()))
        if use_cache and cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # Computar geodésica
        if self.geodesic is not None:
            try:
                geo_path = self.geodesic.shortest_path(
                    start_latent, 
                    end_latent,
                    max_iterations=self.config.path_search_iterations
                )
                
                points = geo_path.points
                geodesic_length = geo_path.length
                converged = geo_path.converged
                
            except Exception as e:
                logger.warning(f"Geodesic computation failed: {e}")
                points, geodesic_length, converged = self._fallback_path(start_latent, end_latent)
        else:
            points, geodesic_length, converged = self._fallback_path(start_latent, end_latent)
        
        # Calcular métricas adicionais
        euclidean_length = np.linalg.norm(end_latent - start_latent)
        curvature_integral = self._compute_path_curvature(points)
        attractors_visited = self._find_attractors_on_path(points)
        
        # Suavizar caminho se configurado
        if self.config.path_smoothing and len(points) > 2:
            points = self._smooth_path(points)
        
        result = SemanticPath(
            start=start_latent,
            end=end_latent,
            points=points,
            geodesic_length=geodesic_length,
            euclidean_length=euclidean_length,
            curvature_integral=curvature_integral,
            attractors_visited=attractors_visited,
            converged=converged
        )
        
        # Cachear
        if use_cache:
            self._path_cache[cache_key] = result
        
        return result
    
    def _fallback_path(
        self, 
        start: np.ndarray, 
        end: np.ndarray,
        n_steps: int = 20
    ) -> Tuple[np.ndarray, float, bool]:
        """Caminho linear como fallback."""
        t = np.linspace(0, 1, n_steps)[:, np.newaxis]
        points = start + t * (end - start)
        length = np.linalg.norm(end - start)
        return points, length, True
    
    def _smooth_path(self, points: np.ndarray) -> np.ndarray:
        """Suaviza caminho via média móvel."""
        alpha = self.config.path_smoothing_factor
        smoothed = points.copy()
        for i in range(1, len(points) - 1):
            smoothed[i] = alpha * points[i-1] + (1 - 2*alpha) * points[i] + alpha * points[i+1]
        return smoothed
    
    def _compute_path_curvature(self, points: np.ndarray) -> float:
        """Calcula integral de curvatura ao longo do caminho."""
        if len(points) < 3:
            return 0.0
        
        total = 0.0
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                total += angle
        
        return total
    
    def _find_attractors_on_path(self, points: np.ndarray) -> List[int]:
        """Encontra atratores visitados pelo caminho."""
        if self.bridge is None:
            return []
        
        visited = []
        for point in points[::max(1, len(points)//10)]:  # Amostrar
            try:
                nearest = self.bridge.get_nearest_anchors(point, k=1)
                if nearest:
                    anchor, dist = nearest[0]
                    if dist < 0.3 and anchor.global_idx not in visited:
                        visited.append(anchor.global_idx)
            except:
                pass
        
        return visited
    
    # =========================================================================
    # PROPAGAÇÃO DE ATIVAÇÃO
    # =========================================================================
    
    def propagate_concept(
        self,
        source: np.ndarray,
        steps: int = None,
        decay: float = None
    ) -> ActivationMap:
        """
        Propaga ativação de um conceito para conceitos relacionados.
        
        Args:
            source: Embedding do conceito fonte
            steps: Número de passos de propagação
            decay: Fator de decaimento por passo
            
        Returns:
            ActivationMap com ativações resultantes
        """
        steps = steps or self.config.propagation_steps
        decay = decay or self.config.propagation_decay
        
        source_latent = self._to_latent(source)
        
        activations = {}
        paths = [] if self.config.store_intermediate_paths else []
        total_energy = 0.0
        
        if self.geodesic is not None and self.bridge is not None:
            try:
                # Usar propagate_activation do GeodesicFlow
                from core.field.manifold import ManifoldPoint
                
                mp = ManifoldPoint(
                    coordinates=source_latent,
                    properties={'activation': 1.0}
                )
                
                result = self.geodesic.propagate_activation(mp, steps, decay)
                
                # Converter resultado para ActivationMap
                for point in result:
                    if hasattr(point, 'properties') and 'activation' in point.properties:
                        act = point.properties['activation']
                        if act > self.config.min_activation_threshold:
                            # Encontrar índice do atrator mais próximo
                            nearest = self.bridge.get_nearest_anchors(point.coordinates, k=1)
                            if nearest:
                                anchor, _ = nearest[0]
                                activations[anchor.global_idx] = max(
                                    activations.get(anchor.global_idx, 0), 
                                    act
                                )
                                total_energy += act
                                
            except Exception as e:
                logger.warning(f"Geodesic propagation failed: {e}")
                activations, total_energy = self._fallback_propagation(source_latent, steps, decay)
        else:
            activations, total_energy = self._fallback_propagation(source_latent, steps, decay)
        
        return ActivationMap(
            source=source_latent,
            activations=activations,
            paths=paths,
            total_energy=total_energy,
            max_radius=steps * self.config.geodesic_step_size * self.config.max_geodesic_steps,
            n_activated=len(activations)
        )
    
    def _fallback_propagation(
        self,
        source: np.ndarray,
        steps: int,
        decay: float
    ) -> Tuple[Dict[int, float], float]:
        """Propagação por vizinhança como fallback."""
        activations = {}
        total_energy = 0.0
        
        if self.bridge is None:
            return activations, total_energy
        
        try:
            # Usar k-nearest anchors como aproximação
            k = min(steps * 5, 50)
            nearest = self.bridge.get_nearest_anchors(source, k=k)
            
            for anchor, dist in nearest:
                activation = np.exp(-dist / (decay + 1e-8))
                if activation > self.config.min_activation_threshold:
                    activations[anchor.global_idx] = activation
                    total_energy += activation
                    
        except Exception as e:
            logger.warning(f"Fallback propagation failed: {e}")
        
        return activations, total_energy
    
    # =========================================================================
    # EXPLORAÇÃO DO CAMPO GEODÉSICO
    # =========================================================================
    
    def geodesic_field(
        self,
        center: np.ndarray,
        n_directions: int = 8
    ) -> GeodesicField:
        """
        Dispara geodésicas em múltiplas direções a partir de um ponto.
        
        Útil para visualizar como o espaço semântico "se abre" a partir de um conceito.
        
        Args:
            center: Ponto central
            n_directions: Número de direções a explorar
            
        Returns:
            GeodesicField com geodésicas em diferentes direções
        """
        center_latent = self._to_latent(center)
        
        # Gerar direções aleatórias uniformes
        dim = len(center_latent)
        directions = np.random.randn(n_directions, dim)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        paths = []
        lengths = []
        curvatures = []
        
        if self.geodesic is not None:
            try:
                for direction in directions:
                    geo_path = self.geodesic.compute_geodesic(
                        center_latent,
                        direction,
                        max_steps=self.config.max_geodesic_steps
                    )
                    paths.append(geo_path.points)
                    lengths.append(geo_path.length)
                    curvatures.append(self._compute_path_curvature(geo_path.points))
                    
            except Exception as e:
                logger.warning(f"Geodesic field computation failed: {e}")
                paths, lengths, curvatures = self._fallback_field(center_latent, directions)
        else:
            paths, lengths, curvatures = self._fallback_field(center_latent, directions)
        
        return GeodesicField(
            center=center_latent,
            directions=directions,
            paths=paths,
            lengths=lengths,
            curvatures=curvatures
        )
    
    def _fallback_field(
        self,
        center: np.ndarray,
        directions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """Campo geodésico linear como fallback."""
        paths = []
        lengths = []
        curvatures = []
        
        step_size = self.config.geodesic_step_size
        n_steps = self.config.max_geodesic_steps
        
        for direction in directions:
            t = np.linspace(0, step_size * n_steps, n_steps)[:, np.newaxis]
            path = center + t * direction
            paths.append(path)
            lengths.append(float(np.linalg.norm(path[-1] - path[0])))
            curvatures.append(0.0)
        
        return paths, lengths, curvatures
    
    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================
    
    def _to_latent(self, embedding: np.ndarray) -> np.ndarray:
        """Converte embedding para espaço latente do bridge."""
        if self.bridge is not None and hasattr(self.bridge, '_project_to_latent'):
            return self.bridge._project_to_latent(embedding)
        return embedding
    
    def distance(self, a: np.ndarray, b: np.ndarray, geodesic: bool = True) -> float:
        """
        Calcula distância entre dois pontos.
        
        Args:
            a, b: Embeddings dos pontos
            geodesic: Se True, usa distância geodésica; caso contrário, euclidiana
            
        Returns:
            Distância entre os pontos
        """
        if geodesic:
            path = self.semantic_path(a, b, use_cache=True)
            return path.geodesic_length
        else:
            return float(np.linalg.norm(self._to_latent(a) - self._to_latent(b)))
    
    def clear_cache(self):
        """Limpa cache de caminhos."""
        self._path_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "has_geodesic_flow": self.geodesic is not None,
            "cached_paths": len(self._path_cache),
            "config": {
                "max_geodesic_steps": self.config.max_geodesic_steps,
                "propagation_decay": self.config.propagation_decay,
                "use_attractor_guidance": self.config.use_attractor_guidance
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_geodesic_bridge(
    bridge,
    **config_kwargs
) -> GeodesicBridgeIntegration:
    """
    Factory function para criar GeodesicBridgeIntegration.
    
    Args:
        bridge: VQVAEManifoldBridge conectado
        **config_kwargs: Parâmetros de configuração
        
    Returns:
        GeodesicBridgeIntegration configurada
    """
    config = GeodesicBridgeConfig(**config_kwargs)
    return GeodesicBridgeIntegration(bridge, config=config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Geodesic ↔ Bridge Integration - Teste")
    print("=" * 60)
    
    # Mock VQ-VAE e Bridge
    class MockVQVAE:
        def get_codebook(self):
            np.random.seed(42)
            return np.random.randn(4, 256, 128).astype(np.float32)
    
    class MockBridge:
        def __init__(self):
            self.latent_dim = 128
            np.random.seed(42)
            self.anchors = [type('Anchor', (), {'global_idx': i, 'coordinates': np.random.randn(128)})() 
                          for i in range(10)]
        
        def _project_to_latent(self, x):
            return x
        
        def get_nearest_anchors(self, point, k=1):
            dists = [(a, np.linalg.norm(point - a.coordinates)) for a in self.anchors]
            dists.sort(key=lambda x: x[1])
            return dists[:k]
    
    # Criar integração
    print("\n1. Criando integração...")
    bridge = MockBridge()
    gbi = GeodesicBridgeIntegration(bridge)
    print(f"   Stats: {gbi.stats()}")
    
    # Teste: Caminho semântico
    print("\n2. Testando caminho semântico...")
    start = np.random.randn(128)
    end = np.random.randn(128)
    path = gbi.semantic_path(start, end)
    print(f"   Geodesic length: {path.geodesic_length:.4f}")
    print(f"   Euclidean length: {path.euclidean_length:.4f}")
    print(f"   Geodesic ratio: {path.geodesic_ratio:.4f}")
    print(f"   Steps: {path.n_steps}")
    print(f"   Attractors visited: {path.attractors_visited}")
    
    # Teste: Propagação
    print("\n3. Testando propagação de conceito...")
    source = np.random.randn(128)
    activations = gbi.propagate_concept(source, steps=3)
    print(f"   Activated: {activations.n_activated}")
    print(f"   Total energy: {activations.total_energy:.4f}")
    print(f"   Top activations: {activations.get_top_activations(3)}")
    
    # Teste: Campo geodésico
    print("\n4. Testando campo geodésico...")
    center = np.random.randn(128)
    field = gbi.geodesic_field(center, n_directions=4)
    print(f"   Directions: {len(field.directions)}")
    print(f"   Mean length: {field.mean_length:.4f}")
    print(f"   Mean curvature: {field.mean_curvature:.4f}")
    
    # Teste: Distância
    print("\n5. Testando distância...")
    a = np.random.randn(128)
    b = np.random.randn(128)
    geo_dist = gbi.distance(a, b, geodesic=True)
    euc_dist = gbi.distance(a, b, geodesic=False)
    print(f"   Geodesic distance: {geo_dist:.4f}")
    print(f"   Euclidean distance: {euc_dist:.4f}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
