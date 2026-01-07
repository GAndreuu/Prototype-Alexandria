"""
Efficient Geodesic Computation via Dimensionality Reduction
============================================================

Resolve o problema de geodésicas proibitivas em 384D usando
redução de dimensão adaptativa.

Estratégia Híbrida:
    1. Treinar PCA em dataset de embeddings
    2. Reduzir 384D → 64D para cálculo geodésico
    3. Computar geodésica no espaço reduzido
    4. Expandir resultado de volta para 384D

Complexidade:
    Original: O(384²) = 147,456 ops/ponto
    Otimizado: O(64²) = 4,096 ops/ponto
    Speedup: 36x

Uso:
    from efficient_geodesic import EfficientGeodesicComputer
    
    computer = EfficientGeodesicComputer()
    computer.fit(embeddings_dataset)  # Train PCA
    
    path = computer.geodesic(start_384, target_384)
    print(f"Path length: {len(path.points)}")

Autor: G (Alexandria Project)
Versão: 1.0
Status: Move 1 - Challenger Strategy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class EfficientGeodesicConfig:
    """Configuração da geodésica eficiente."""
    
    # Dimensões
    full_dim: int = 384              # Dimensão original (embeddings)
    reduced_dim: int = 64            # Dimensão para geodésica
    
    # PCA
    pca_variance_threshold: float = 0.95  # Variância explicada mínima
    pca_whiten: bool = False         # Whitening (não recomendado para geodésicas)
    
    # Geodésica
    n_steps: int = 20                # Passos na geodésica
    step_size: float = 0.1           # Tamanho do passo
    convergence_threshold: float = 0.001  # Para early stopping
    max_iterations: int = 100        # Iterações para refinamento
    
    # Métrica
    use_learned_metric: bool = False  # Usar métrica aprendida vs Euclidiana
    metric_regularization: float = 0.01  # Regularização da métrica
    
    # Performance
    batch_size: int = 32             # Para processamento em lote
    cache_pca: bool = True           # Cache de componentes PCA


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class GeodesicPath:
    """Resultado de uma computação geodésica."""
    
    # Caminho
    points: np.ndarray               # [n_steps, full_dim] pontos no espaço original
    points_reduced: np.ndarray       # [n_steps, reduced_dim] pontos no espaço reduzido
    velocities: np.ndarray           # [n_steps, full_dim] velocidades
    
    # Métricas
    length: float                    # Comprimento total
    length_reduced: float            # Comprimento no espaço reduzido
    n_steps: int
    converged: bool
    
    # Curvatura
    curvatures: np.ndarray           # Curvatura em cada ponto
    mean_curvature: float
    max_curvature: float
    
    # Energia
    energy: float                    # Energia da geodésica (integral de |v|²)
    
    # Performance
    computation_time_ms: float = 0.0
    pca_reconstruction_error: float = 0.0  # Erro ao expandir de volta
    
    def summary(self) -> str:
        """Resumo da geodésica."""
        return (
            f"GeodesicPath: {self.n_steps} steps, "
            f"length={self.length:.4f}, "
            f"curvature_mean={self.mean_curvature:.4f}, "
            f"converged={self.converged}, "
            f"time={self.computation_time_ms:.1f}ms"
        )


@dataclass
class MetricTensor:
    """Tensor métrico local."""
    
    position: np.ndarray             # Onde a métrica é definida
    g: np.ndarray                    # [dim, dim] matriz métrica
    g_inv: np.ndarray                # [dim, dim] inversa
    christoffel: Optional[np.ndarray] = None  # [dim, dim, dim] símbolos


# =============================================================================
# COMPUTADOR DE GEODÉSICA EFICIENTE
# =============================================================================

class EfficientGeodesicComputer:
    """
    Computa geodésicas de forma eficiente via redução dimensional.
    
    A ideia central é que a estrutura geodésica em 384D é bem aproximada
    pela estrutura em 64D (onde 64D captura ~90-95% da variância).
    
    Isso permite computar geodésicas reais em tempo viável.
    """
    
    def __init__(self, config: Optional[EfficientGeodesicConfig] = None):
        self.config = config or EfficientGeodesicConfig()
        
        # PCA para redução
        self._pca: Optional[PCA] = None
        self._is_fitted = False
        
        # Cache de métricas
        self._metric_cache: Dict[str, MetricTensor] = {}
        
        # Estatísticas
        self._n_geodesics_computed = 0
        self._total_time_ms = 0.0
        
        logger.info(
            f"EfficientGeodesicComputer: {self.config.full_dim}D → "
            f"{self.config.reduced_dim}D (speedup: "
            f"{(self.config.full_dim**2) // (self.config.reduced_dim**2)}x)"
        )
    
    # =========================================================================
    # FITTING (PCA)
    # =========================================================================
    
    def fit(self, embeddings: np.ndarray, variance_threshold: Optional[float] = None):
        """
        Treina PCA em dataset de embeddings.
        
        Args:
            embeddings: [n_samples, full_dim] matriz de embeddings
            variance_threshold: Override do threshold de variância
            
        Importante:
            - Chamar antes de usar geodesic()
            - Usar embeddings representativos do domínio
            - Pode re-treinar se domínio mudar
        """
        logger.info(f"Fitting PCA on {embeddings.shape[0]} embeddings...")
        
        threshold = variance_threshold or self.config.pca_variance_threshold
        
        # Determinar número de componentes para threshold
        pca_full = PCA(n_components=min(embeddings.shape[0], embeddings.shape[1]))
        pca_full.fit(embeddings)
        
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumvar >= threshold) + 1
        n_components = max(n_components, self.config.reduced_dim)
        n_components = min(n_components, self.config.reduced_dim)  # Cap at reduced_dim
        
        # Re-fit com número correto de componentes
        self._pca = PCA(
            n_components=n_components,
            whiten=self.config.pca_whiten
        )
        self._pca.fit(embeddings)
        
        self._is_fitted = True
        
        explained = np.sum(self._pca.explained_variance_ratio_)
        logger.info(
            f"PCA fitted: {n_components} components, "
            f"{explained*100:.1f}% variance explained"
        )
        
        return self
    
    def fit_from_manifold(self, manifold):
        """
        Treina PCA a partir de pontos já na variedade.
        
        Args:
            manifold: DynamicManifold com pontos
        """
        if len(manifold.points) < 10:
            logger.warning("Few points in manifold, using random initialization")
            # Criar embeddings sintéticos
            embeddings = np.random.randn(100, self.config.full_dim)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            embeddings = manifold.get_coordinates_matrix()
            # Truncar para dimensão base se expandido
            if embeddings.shape[1] > self.config.full_dim:
                embeddings = embeddings[:, :self.config.full_dim]
        
        return self.fit(embeddings)
    
    # =========================================================================
    # PROJEÇÃO
    # =========================================================================
    
    def reduce(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reduz embedding de full_dim para reduced_dim.
        
        Args:
            embedding: [full_dim] ou [n, full_dim]
            
        Returns:
            [reduced_dim] ou [n, reduced_dim]
        """
        if not self._is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        is_1d = embedding.ndim == 1
        if is_1d:
            embedding = embedding.reshape(1, -1)
        
        # Truncar se necessário
        if embedding.shape[1] > self.config.full_dim:
            embedding = embedding[:, :self.config.full_dim]
        
        reduced = self._pca.transform(embedding)
        
        return reduced[0] if is_1d else reduced
    
    def expand(self, reduced: np.ndarray) -> np.ndarray:
        """
        Expande de reduced_dim de volta para full_dim.
        
        Args:
            reduced: [reduced_dim] ou [n, reduced_dim]
            
        Returns:
            [full_dim] ou [n, full_dim]
        """
        if not self._is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        is_1d = reduced.ndim == 1
        if is_1d:
            reduced = reduced.reshape(1, -1)
        
        expanded = self._pca.inverse_transform(reduced)
        
        return expanded[0] if is_1d else expanded
    
    def reconstruction_error(self, embedding: np.ndarray) -> float:
        """
        Calcula erro de reconstrução (quanto se perde na redução).
        
        Args:
            embedding: [full_dim]
            
        Returns:
            Erro médio quadrático de reconstrução
        """
        reduced = self.reduce(embedding)
        reconstructed = self.expand(reduced)
        return float(np.mean((embedding[:self.config.full_dim] - reconstructed) ** 2))
    
    # =========================================================================
    # GEODÉSICA PRINCIPAL
    # =========================================================================
    
    def geodesic(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_steps: Optional[int] = None
    ) -> GeodesicPath:
        """
        Computa geodésica entre dois pontos.
        
        Estratégia:
            1. Reduzir start e target para 64D
            2. Computar geodésica no espaço reduzido
            3. Expandir cada ponto de volta para 384D
        
        Args:
            start: [full_dim] ponto inicial
            target: [full_dim] ponto final
            n_steps: Override do número de passos
            
        Returns:
            GeodesicPath com o caminho completo
        """
        t0 = time.time()
        n_steps = n_steps or self.config.n_steps
        
        # Auto-fit se necessário
        if not self._is_fitted:
            logger.warning("PCA not fitted, using random initialization")
            synthetic = np.vstack([
                start.reshape(1, -1),
                target.reshape(1, -1),
                np.random.randn(98, self.config.full_dim)
            ])
            self.fit(synthetic)
        
        # 1. Reduzir para espaço de baixa dimensão
        start_reduced = self.reduce(start)
        target_reduced = self.reduce(target)
        
        # 2. Computar geodésica no espaço reduzido
        path_reduced, velocities_reduced, curvatures = self._compute_geodesic_reduced(
            start_reduced, target_reduced, n_steps
        )
        
        # 3. Expandir de volta para espaço original
        path_full = self.expand(path_reduced)
        
        # Velocidades no espaço original (diferenças finitas)
        velocities_full = np.zeros_like(path_full)
        velocities_full[:-1] = np.diff(path_full, axis=0)
        velocities_full[-1] = velocities_full[-2]
        
        # Métricas
        length_reduced = self._path_length(path_reduced)
        length_full = self._path_length(path_full)
        energy = self._path_energy(velocities_full)
        
        # Erro de reconstrução
        recon_error = 0.5 * (
            self.reconstruction_error(start) + 
            self.reconstruction_error(target)
        )
        
        elapsed_ms = (time.time() - t0) * 1000
        self._n_geodesics_computed += 1
        self._total_time_ms += elapsed_ms
        
        return GeodesicPath(
            points=path_full,
            points_reduced=path_reduced,
            velocities=velocities_full,
            length=length_full,
            length_reduced=length_reduced,
            n_steps=n_steps,
            converged=True,  # TODO: verificar convergência real
            curvatures=curvatures,
            mean_curvature=float(np.mean(np.abs(curvatures))),
            max_curvature=float(np.max(np.abs(curvatures))),
            energy=energy,
            computation_time_ms=elapsed_ms,
            pca_reconstruction_error=recon_error
        )
    
    def _compute_geodesic_reduced(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computa geodésica no espaço reduzido.
        
        Para métrica Euclidiana, geodésica = linha reta.
        Para métrica aprendida, usa relaxation.
        
        Returns:
            (points, velocities, curvatures)
        """
        dim = len(start)
        path = np.zeros((n_steps, dim))
        velocities = np.zeros((n_steps, dim))
        curvatures = np.zeros(n_steps)
        
        if self.config.use_learned_metric:
            # Geodésica com métrica aprendida (mais caro, mais preciso)
            path, velocities, curvatures = self._geodesic_with_metric(
                start, target, n_steps
            )
        else:
            # Geodésica Euclidiana (linha reta interpolada)
            for i in range(n_steps):
                t = i / (n_steps - 1)
                path[i] = start + t * (target - start)
                velocities[i] = (target - start) / n_steps
            
            # Curvatura = 0 para linha reta (mas estimar do espaço)
            curvatures = self._estimate_curvature_along_path(path)
        
        return path, velocities, curvatures
    
    def _geodesic_with_metric(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Geodésica usando métrica local aprendida.
        
        Usa relaxation: começa com linha reta, ajusta iterativamente
        para minimizar comprimento local.
        """
        dim = len(start)
        
        # Inicializa com linha reta
        path = np.zeros((n_steps, dim))
        for i in range(n_steps):
            t = i / (n_steps - 1)
            path[i] = start + t * (target - start)
        
        # Relaxation iterativa
        for iteration in range(self.config.max_iterations):
            old_path = path.copy()
            
            # Ajusta pontos internos
            for i in range(1, n_steps - 1):
                # Métrica local
                metric = self._get_metric_at(path[i])
                
                # Direção de geodésica (minimiza curvatura)
                prev_dir = path[i] - path[i-1]
                next_dir = path[i+1] - path[i]
                
                # Ajuste baseado na métrica
                correction = self._geodesic_correction(
                    path[i], prev_dir, next_dir, metric
                )
                
                path[i] += self.config.step_size * correction
            
            # Verificar convergência
            change = np.max(np.abs(path - old_path))
            if change < self.config.convergence_threshold:
                break
        
        # Computar velocidades e curvaturas finais
        velocities = np.zeros((n_steps, dim))
        velocities[:-1] = np.diff(path, axis=0)
        velocities[-1] = velocities[-2]
        
        curvatures = self._estimate_curvature_along_path(path)
        
        return path, velocities, curvatures
    
    def _get_metric_at(self, point: np.ndarray) -> MetricTensor:
        """
        Obtém tensor métrico em um ponto.
        
        Por agora, usa identidade + regularização.
        TODO: Aprender métrica de dados.
        """
        point_key = hash(point.tobytes())
        
        if point_key in self._metric_cache:
            return self._metric_cache[point_key]
        
        dim = len(point)
        
        # Métrica = Identidade + regularização
        g = np.eye(dim) + self.config.metric_regularization * np.random.randn(dim, dim)
        g = (g + g.T) / 2  # Simetrizar
        g += np.eye(dim) * 0.1  # Garantir positiva definida
        
        try:
            g_inv = np.linalg.inv(g)
        except:
            g_inv = np.eye(dim)
        
        metric = MetricTensor(
            position=point.copy(),
            g=g,
            g_inv=g_inv
        )
        
        if self.config.cache_pca:
            self._metric_cache[point_key] = metric
        
        return metric
    
    def _geodesic_correction(
        self,
        point: np.ndarray,
        prev_dir: np.ndarray,
        next_dir: np.ndarray,
        metric: MetricTensor
    ) -> np.ndarray:
        """
        Correção para ponto interno da geodésica.
        
        Minimiza curvatura local (segunda derivada).
        """
        # Aceleração = segunda derivada discreta
        acceleration = next_dir - prev_dir
        
        # Correção = -acceleration projetada na métrica
        correction = -np.dot(metric.g_inv, acceleration)
        
        # Limitar magnitude
        norm = np.linalg.norm(correction)
        if norm > 0.1:
            correction = 0.1 * correction / norm
        
        return correction
    
    def _estimate_curvature_along_path(self, path: np.ndarray) -> np.ndarray:
        """
        Estima curvatura ao longo do caminho.
        
        Curvatura ≈ |segunda derivada| / |primeira derivada|²
        """
        n_steps = len(path)
        curvatures = np.zeros(n_steps)
        
        if n_steps < 3:
            return curvatures
        
        for i in range(1, n_steps - 1):
            # Primeira derivada (velocidade)
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            # Segunda derivada (aceleração)
            a = v2 - v1
            
            # Curvatura = |a| / |v|²
            v_norm = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
            if v_norm > 1e-8:
                curvatures[i] = np.linalg.norm(a) / (v_norm ** 2)
        
        # Extrapolar bordas
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures
    
    def _path_length(self, path: np.ndarray) -> float:
        """Comprimento do caminho."""
        if len(path) < 2:
            return 0.0
        segments = np.diff(path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return float(np.sum(lengths))
    
    def _path_energy(self, velocities: np.ndarray) -> float:
        """Energia do caminho (integral de |v|²)."""
        return float(np.sum(np.linalg.norm(velocities, axis=1) ** 2))
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def geodesic_batch(
        self,
        starts: np.ndarray,
        targets: np.ndarray,
        n_steps: Optional[int] = None
    ) -> List[GeodesicPath]:
        """
        Computa múltiplas geodésicas em batch.
        
        Args:
            starts: [n, full_dim]
            targets: [n, full_dim]
            n_steps: Passos por geodésica
            
        Returns:
            Lista de GeodesicPath
        """
        n = len(starts)
        results = []
        
        for i in range(n):
            path = self.geodesic(starts[i], targets[i], n_steps)
            results.append(path)
        
        return results
    
    # =========================================================================
    # INTEGRAÇÃO COM COMPOSITIONAL REASONER
    # =========================================================================
    
    def as_flow_interface(self):
        """
        Retorna interface compatível com BridgedGeodesicFlow.
        
        Permite usar EfficientGeodesicComputer onde BridgedGeodesicFlow é esperado.
        """
        return _FlowInterface(self)
    
    # =========================================================================
    # ESTATÍSTICAS
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas de uso."""
        return {
            'geodesics_computed': self._n_geodesics_computed,
            'total_time_ms': self._total_time_ms,
            'avg_time_ms': (
                self._total_time_ms / self._n_geodesics_computed 
                if self._n_geodesics_computed > 0 else 0
            ),
            'is_fitted': self._is_fitted,
            'reduced_dim': self.config.reduced_dim,
            'full_dim': self.config.full_dim,
            'speedup': (self.config.full_dim ** 2) // (self.config.reduced_dim ** 2),
            'pca_variance_explained': (
                float(np.sum(self._pca.explained_variance_ratio_)) 
                if self._is_fitted else 0
            ),
            'metric_cache_size': len(self._metric_cache)
        }


# =============================================================================
# INTERFACE DE COMPATIBILIDADE
# =============================================================================

class _FlowInterface:
    """
    Interface de compatibilidade com BridgedGeodesicFlow.
    
    Permite que EfficientGeodesicComputer seja usado onde
    geodesic_flow é esperado.
    """
    
    def __init__(self, computer: EfficientGeodesicComputer):
        self._computer = computer
    
    def shortest_path(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_iterations: int = 5
    ):
        """
        Interface compatível com BridgedGeodesicFlow.shortest_path().
        """
        n_steps = max(10, n_iterations * 4)
        path = self._computer.geodesic(start, target, n_steps)
        
        # Retornar objeto compatível
        return _PathResult(path)


@dataclass
class _PathResult:
    """Resultado compatível com interface antiga."""
    
    _geodesic_path: GeodesicPath
    
    @property
    def points(self) -> np.ndarray:
        return self._geodesic_path.points
    
    @property
    def velocities(self) -> np.ndarray:
        return self._geodesic_path.velocities
    
    @property
    def length(self) -> float:
        return self._geodesic_path.length


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_efficient_geodesic(
    reduced_dim: int = 64,
    full_dim: int = 384,
    embeddings: Optional[np.ndarray] = None
) -> EfficientGeodesicComputer:
    """
    Factory para criar EfficientGeodesicComputer configurado.
    
    Args:
        reduced_dim: Dimensão reduzida (64 recomendado)
        full_dim: Dimensão original (384 para sentence-transformers)
        embeddings: Dataset para treinar PCA (opcional)
        
    Returns:
        EfficientGeodesicComputer pronto para uso
    """
    config = EfficientGeodesicConfig(
        full_dim=full_dim,
        reduced_dim=reduced_dim
    )
    
    computer = EfficientGeodesicComputer(config)
    
    if embeddings is not None:
        computer.fit(embeddings)
    
    return computer
