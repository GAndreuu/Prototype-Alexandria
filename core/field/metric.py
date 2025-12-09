"""
RiemannianMetric: Métrica Dinâmica que Deforma
==============================================

A métrica define como medir distâncias no espaço. Quando um conceito
é triggerado, a métrica deforma localmente - criando "poços" de atração
e alterando os caminhos geodésicos.

Conceitos-chave:
- Métrica Euclidiana: g_ij = δ_ij (identidade) - espaço "plano"
- Deformação: g_ij(x) varia com posição - espaço "curvo"
- Curvatura: emerge da variação da métrica
- Geodésicas: caminhos de menor distância (afetados pela curvatura)

A "mágica" acontece aqui: triggerar um conceito deforma o espaço,
fazendo conceitos relacionados ficarem "mais pertos" em termos
de distância geodésica, mesmo que a distância Euclidiana não mude.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.sparse import csr_matrix, lil_matrix
from scipy.ndimage import gaussian_filter

from .manifold import DynamicManifold, ManifoldPoint


@dataclass
class MetricConfig:
    """Configuração da métrica."""
    deformation_radius: float = 0.3    # Raio de influência da deformação
    deformation_strength: float = 0.5  # Força máxima da deformação
    decay_rate: float = 0.05           # Taxa de relaxamento da métrica
    min_curvature: float = 0.0         # Curvatura mínima (plano)
    max_curvature: float = 10.0        # Curvatura máxima
    grid_resolution: int = 32          # Resolução do grid para campo


@dataclass
class DeformationEvent:
    """Registro de uma deformação."""
    center: np.ndarray       # Ponto central da deformação
    intensity: float         # Intensidade
    radius: float           # Raio de influência
    timestamp: float        # Quando ocorreu


class RiemannianMetric:
    """
    Métrica Riemanniana dinâmica sobre a variedade.
    
    Em vez de armazenar tensor métrico completo (impraticável em alta dimensão),
    usamos uma representação baseada em deformações locais.
    
    A métrica é definida como:
        g(x) = I + Σ_i D_i(x)
        
    Onde I é identidade e D_i são deformações locais gaussianas
    centradas em pontos ativos.
    
    Attributes:
        manifold: Variedade base
        config: Configuração
        deformations: Lista de deformações ativas
        _curvature_cache: Cache do tensor de curvatura (grid)
    """
    
    def __init__(self, manifold: DynamicManifold, config: Optional[MetricConfig] = None):
        self.manifold = manifold
        self.config = config or MetricConfig()
        self.deformations: List[DeformationEvent] = []
        self._curvature_cache: Optional[np.ndarray] = None
        self._cache_valid = False
        
    # =========================================================================
    # TENSOR MÉTRICO
    # =========================================================================
    
    def metric_at(self, point: np.ndarray) -> np.ndarray:
        """
        Computa tensor métrico g_ij em um ponto específico.
        
        Args:
            point: Coordenadas [dim]
            
        Returns:
            Tensor métrico [dim, dim] (simétrico, positivo-definido)
            
        Nota: Para alta dimensão, isso é caro. Usar com parcimônia.
        
        TODO:
            - Aproximação esparsa para alta dimensão
            - Cache para pontos frequentemente acessados
        """
        dim = len(point)
        
        # Começa com métrica Euclidiana (identidade)
        g = np.eye(dim)
        
        # Adiciona deformações
        for deform in self.deformations:
            # Distância ao centro da deformação
            dist = np.linalg.norm(point - deform.center)
            
            if dist < deform.radius * 3:  # Corte para eficiência
                # Deformação Gaussiana
                weight = deform.intensity * np.exp(-0.5 * (dist / deform.radius) ** 2)
                
                # Deforma a métrica na direção do centro
                # Isso "encurta" distâncias na direção da deformação
                direction = (deform.center - point)
                if np.linalg.norm(direction) > 1e-8:
                    direction = direction / np.linalg.norm(direction)
                    # Tensor de deformação: aumenta "peso" na direção do centro
                    D = weight * np.outer(direction, direction)
                    g = g + D
                    
        return g
    
    def metric_tensor_inverse(self, point: np.ndarray) -> np.ndarray:
        """
        Computa inverso do tensor métrico g^ij.
        
        Necessário para levantar índices e computar geodésicas.
        """
        g = self.metric_at(point)
        return np.linalg.inv(g)
    
    def distance(self, p1: np.ndarray, p2: np.ndarray, steps: int = 10) -> float:
        """
        Computa distância aproximada entre dois pontos.
        
        A distância Riemanniana é a integral da métrica ao longo
        do caminho. Aqui aproximamos com soma discreta.
        
        Args:
            p1, p2: Pontos [dim]
            steps: Passos para discretização
            
        Returns:
            Distância aproximada
            
        TODO:
            - Usar geodésica real em vez de linha reta
            - Integração adaptativa
        """
        # Interpola ao longo da linha reta (aproximação)
        t = np.linspace(0, 1, steps)
        total_dist = 0.0
        
        for i in range(steps - 1):
            # Ponto intermediário
            pt = p1 + t[i] * (p2 - p1)
            # Vetor tangente
            v = (p2 - p1) / steps
            # Métrica nesse ponto
            g = self.metric_at(pt)
            # Comprimento: sqrt(v^T g v)
            ds = np.sqrt(np.dot(v, np.dot(g, v)))
            total_dist += ds
            
        return total_dist
    
    # =========================================================================
    # DEFORMAÇÃO
    # =========================================================================
    
    def deform_at(self, point: np.ndarray, intensity: Optional[float] = None, 
                  radius: Optional[float] = None):
        """
        Cria deformação centrada em um ponto.
        
        Isso é chamado quando um conceito é "triggerado".
        A deformação cria um "poço" no espaço que atrai geodésicas.
        
        Args:
            point: Centro da deformação [dim]
            intensity: Força da deformação (default: config)
            radius: Raio de influência (default: config)
        """
        intensity = intensity or self.config.deformation_strength
        radius = radius or self.config.deformation_radius
        
        event = DeformationEvent(
            center=point.copy(),
            intensity=intensity,
            radius=radius,
            timestamp=len(self.deformations)  # Simplificado
        )
        
        self.deformations.append(event)
        self._cache_valid = False
        
    def deform_at_point(self, manifold_point: ManifoldPoint):
        """
        Deforma métrica baseado em ManifoldPoint com sua ativação.
        """
        intensity = manifold_point.activation * self.config.deformation_strength
        self.deform_at(manifold_point.coordinates, intensity=intensity)
        
    def relax(self, rate: Optional[float] = None):
        """
        Relaxa deformações (decaimento temporal).
        
        A métrica gradualmente retorna ao estado plano.
        """
        rate = rate or self.config.decay_rate
        
        surviving = []
        for d in self.deformations:
            d.intensity *= (1 - rate)
            if d.intensity > 0.01:  # Threshold
                surviving.append(d)
                
        self.deformations = surviving
        self._cache_valid = False
        
    def clear_deformations(self):
        """Remove todas as deformações."""
        self.deformations = []
        self._cache_valid = False
        
    # =========================================================================
    # CURVATURA
    # =========================================================================
    
    def curvature_scalar_at(self, point: np.ndarray, epsilon: float = 0.01) -> float:
        """
        Computa curvatura escalar (Ricci scalar) em um ponto.
        
        Curvatura alta = região muito deformada = atrator forte
        Curvatura baixa/zero = região plana = sem atração especial
        
        Args:
            point: Coordenadas [dim]
            epsilon: Step para diferenças finitas
            
        Returns:
            Curvatura escalar (pode ser positiva ou negativa)
            
        TODO:
            - Implementação real via símbolos de Christoffel
            - Por agora, aproximação via Laplaciano da métrica
        """
        # Aproximação: curvatura ~ Laplaciano do determinante da métrica
        # Isso não é rigorosamente correto mas captura o comportamento qualitativo
        
        dim = len(point)
        
        # Determinante da métrica no ponto
        g_center = self.metric_at(point)
        det_center = np.linalg.det(g_center)
        
        # Laplaciano via diferenças finitas
        laplacian = 0.0
        for i in range(min(dim, 10)):  # Limita para eficiência
            # Derivada segunda na direção i
            point_plus = point.copy()
            point_plus[i] += epsilon
            point_minus = point.copy()
            point_minus[i] -= epsilon
            
            det_plus = np.linalg.det(self.metric_at(point_plus))
            det_minus = np.linalg.det(self.metric_at(point_minus))
            
            d2_det = (det_plus - 2*det_center + det_minus) / (epsilon**2)
            laplacian += d2_det
            
        # Curvatura proporcional ao Laplaciano (com sinal)
        curvature = -laplacian / (det_center + 1e-8)
        
        # Clipa para range razoável
        return np.clip(curvature, -self.config.max_curvature, self.config.max_curvature)
    
    def curvature_tensor(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computa campo de curvatura em grid de pontos.
        
        Args:
            grid_points: [n_points, dim] ou None para usar pontos da variedade
            
        Returns:
            Array de curvaturas [n_points]
            
        Este é o "campo" que visualizamos - a paisagem de curvatura.
        """
        if grid_points is None:
            grid_points = self.manifold.get_coordinates_matrix()
            
        if len(grid_points) == 0:
            return np.array([])
            
        curvatures = np.array([
            self.curvature_scalar_at(p) for p in grid_points
        ])
        
        return curvatures
    
    # =========================================================================
    # SÍMBOLOS DE CHRISTOFFEL (para geodésicas)
    # =========================================================================
    
    def christoffel_at(self, point: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        Computa símbolos de Christoffel Γ^k_ij em um ponto.
        
        Estes são necessários para a equação geodésica:
        d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
        
        Args:
            point: Coordenadas [dim]
            epsilon: Step para diferenças finitas
            
        Returns:
            Tensor [dim, dim, dim] com Γ^k_ij
            
        TODO:
            - Isso é O(dim³) - precisa de aproximação esparsa
            - Cache para pontos frequentes
        """
        dim = len(point)
        
        # Para alta dimensão, usar versão esparsa
        if dim > 50:
            return self._christoffel_sparse(point, epsilon)
            
        gamma = np.zeros((dim, dim, dim))
        
        # g^{kl} (inverso da métrica)
        g_inv = self.metric_tensor_inverse(point)
        
        # Derivadas da métrica
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
                    
                    sum_term = 0.0
                    for l in range(dim):
                        # ∂_i g_{jl}
                        dg_i = self._metric_derivative(point, i, j, l, epsilon)
                        # ∂_j g_{il}
                        dg_j = self._metric_derivative(point, j, i, l, epsilon)
                        # ∂_l g_{ij}
                        dg_l = self._metric_derivative(point, l, i, j, epsilon)
                        
                        sum_term += g_inv[k, l] * (dg_i + dg_j - dg_l)
                        
                    gamma[k, i, j] = 0.5 * sum_term
                    
        return gamma
    
    def _christoffel_sparse(self, point: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Versão esparsa do cálculo de Christoffel.
        
        Só computa para direções com deformação significativa.
        
        TODO: Implementar
        """
        dim = len(point)
        # Por agora, retorna zeros (métrica plana)
        return np.zeros((dim, dim, dim))
    
    def _metric_derivative(self, point: np.ndarray, direction: int, 
                          i: int, j: int, epsilon: float) -> float:
        """
        Derivada parcial da componente g_ij na direção especificada.
        
        ∂_direction g_ij
        """
        point_plus = point.copy()
        point_plus[direction] += epsilon
        point_minus = point.copy()
        point_minus[direction] -= epsilon
        
        g_plus = self.metric_at(point_plus)
        g_minus = self.metric_at(point_minus)
        
        return (g_plus[i, j] - g_minus[i, j]) / (2 * epsilon)
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def find_wells(self, threshold: float = 0.5) -> List[np.ndarray]:
        """
        Encontra "poços" de curvatura (mínimos locais de energia).
        
        Estes são os atratores - pontos onde geodésicas tendem a convergir.
        
        Args:
            threshold: Curvatura mínima para considerar como poço
            
        Returns:
            Lista de coordenadas dos poços
            
        TODO:
            - Algoritmo de watershed ou gradient descent
            - Por agora, retorna centros das deformações fortes
        """
        wells = []
        for d in self.deformations:
            if d.intensity > threshold:
                wells.append(d.center)
        return wells
    
    def gradient_at(self, point: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        Gradiente da curvatura em um ponto.
        
        Indica direção de maior aumento de curvatura.
        """
        dim = len(point)
        grad = np.zeros(dim)
        
        curv_center = self.curvature_scalar_at(point, epsilon)
        
        for i in range(min(dim, 20)):  # Limita para eficiência
            point_plus = point.copy()
            point_plus[i] += epsilon
            curv_plus = self.curvature_scalar_at(point_plus, epsilon)
            grad[i] = (curv_plus - curv_center) / epsilon
            
        return grad
    
    def stats(self) -> Dict:
        """Estatísticas da métrica."""
        return {
            'num_deformations': len(self.deformations),
            'total_intensity': sum(d.intensity for d in self.deformations),
            'mean_radius': np.mean([d.radius for d in self.deformations]) if self.deformations else 0,
            'num_wells': len(self.find_wells()),
        }
