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
# from scipy.sparse import csr_matrix, lil_matrix
# from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
import os

from .manifold import DynamicManifold, ManifoldPoint


@dataclass
class MetricConfig:
    """Configuração da métrica."""
    deformation_radius: float = 0.8         # Raio de influência (aumentado de 0.3)
    deformation_strength: float = 1.0       # Força máxima da deformação (aumentado)
    decay_rate: float = 0.05                # Taxa de relaxamento da métrica
    min_curvature: float = 0.0              # Curvatura mínima (plano)
    max_curvature: float = 10.0             # Curvatura máxima
    grid_resolution: int = 32               # Resolução do grid para campo
    
    # Novos parâmetros para comportamento melhorado
    kernel_type: str = "cauchy"             # "gaussian" | "cauchy" (cauchy decai mais lento)
    mode: str = "attractor"                 # "attractor" | "obstacle"
                                            # attractor: g = I + w*(I - n⊗n) → favorece caminhos pelo centro
                                            # obstacle: g = I + w*(n⊗n) → repele caminhos
    cutoff_multiplier: float = 6.0          # Cutoff = radius * multiplier (aumentado de 3)
    radius_scale_with_dim: bool = True      # Escala raio com sqrt(dim/10)


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
        
        Fórmula:
        - mode="attractor": g = I + Σ w_k * (I - n_k⊗n_k)  → favorece caminhos pelo centro
        - mode="obstacle":  g = I + Σ w_k * (n_k⊗n_k)     → repele caminhos
        
        Kernel:
        - "gaussian": w = intensity * exp(-0.5 * (dist/radius)²)
        - "cauchy":   w = intensity / (1 + (dist/radius)²)
        
        Args:
            point: Coordenadas [dim]
            
        Returns:
            Tensor métrico [dim, dim] (simétrico, positivo-definido)
        """
        dim = len(point)
        g = np.eye(dim)
        
        if not self.deformations:
            return g
            
        # Preparar arrays das deformações
        centers = np.array([d.center for d in self.deformations]) # [N, dim]
        intensities = np.array([d.intensity for d in self.deformations]) # [N]
        radii = np.array([d.radius for d in self.deformations]) # [N]
        
        # Scaling radius logic
        if self.config.radius_scale_with_dim:
             radii = radii * np.sqrt(dim / 10.0)
             
        # Cutoff logic
        base_radius = self.config.deformation_radius
        if self.config.radius_scale_with_dim:
            effective_radius = base_radius * np.sqrt(dim / 10.0)
        else:
            effective_radius = base_radius
            
        cutoff = effective_radius * self.config.cutoff_multiplier
        
        # Distances
        diffs = centers - point[None, :] # [N, dim]
        dist_sq = np.sum(diffs**2, axis=1) # [N]
        
        # Mask
        mask = (dist_sq > 1e-12) & (dist_sq < cutoff**2)
        
        if not np.any(mask):
            return g
            
        # Filter active
        diffs = diffs[mask]
        dist_sq = dist_sq[mask]
        intensities = intensities[mask]
        radii = radii[mask]
        
        # Weights
        r_sq = radii**2 + 1e-12
        if self.config.kernel_type == "cauchy":
            weights = intensities / (1.0 + dist_sq / r_sq)
        else:
            weights = intensities * np.exp(-0.5 * dist_sq / r_sq)
            
        # Directions n
        inv_dist = 1.0 / np.sqrt(dist_sq + 1e-12)
        n = diffs * inv_dist[:, None] # [K, dim]
        
        # Accumulate tensors
        if self.config.mode == "attractor":
            # g = (1 + sum(w))I - sum(w * n nT)
            total_weight = np.sum(weights)
            g *= (1.0 + total_weight)
            
            wn = n * weights[:, None]
            g -= n.T @ wn
        else:
            # g = I + sum(w * n nT)
            wn = n * weights[:, None]
            g += n.T @ wn
            
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
        
        # Determinante da métrica no ponto (usando log-det para estabilidade)
        g_center = self.metric_at(point)
        sign_center, logdet_center = np.linalg.slogdet(g_center)
        
        # Laplaciano do log-det via diferenças finitas
        laplacian_log = 0.0
        for i in range(min(dim, 10)):  # Limita para eficiência
            # Derivada segunda na direção i
            point_plus = point.copy()
            point_plus[i] += epsilon
            point_minus = point.copy()
            point_minus[i] -= epsilon
            
            sign_plus, logdet_plus = np.linalg.slogdet(self.metric_at(point_plus))
            sign_minus, logdet_minus = np.linalg.slogdet(self.metric_at(point_minus))
            
            # d²(log det g) / dx²
            d2_log = (logdet_plus - 2*logdet_center + logdet_minus) / (epsilon**2)
            laplacian_log += d2_log
            
        # Curvatura aproximada como negativo do Laplaciano do log-volume
        # R ~ -Δ(log √|g|) = -0.5 * Δ(log |g|)
        curvature = -0.5 * laplacian_log
        
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
    
    def christoffel_at(self, point: np.ndarray, epsilon: float = 0.01, n_workers: int = 1) -> np.ndarray:
        """
        Computa símbolos de Christoffel Γ^k_ij em um ponto.
        
        Estes são necessários para a equação geodésica:
        d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
        
        Args:
            point: Coordenadas [dim]
            epsilon: Step para diferenças finitas
            n_workers: Número de workers para paralelização (default: 1)
            
        Returns:
            Tensor [dim, dim, dim] com Γ^k_ij
        """
        dim = len(point)
        
        # Para alta dimensão, usar versão esparsa
        if dim > 50:
            return self._christoffel_sparse(point, epsilon)
            
        gamma = np.zeros((dim, dim, dim))
        
        # g^{kl} (inverso da métrica)
        g_inv = self.metric_tensor_inverse(point)
        
        # Pré-computa todas as derivadas da métrica (o gargalo)
        # Isso evita recomputar a mesma derivada múltiplas vezes
        deriv_cache = {}
        
        def compute_derivative(args):
            direction, i, j = args
            key = (direction, i, j)
            return key, self._metric_derivative(point, direction, i, j, epsilon)
        
        # Gera todas as combinações únicas de derivadas necessárias
        deriv_args = []
        for direction in range(dim):
            for i in range(dim):
                for j in range(i, dim + 1):  # j >= i para simetria
                    if j < dim:
                        deriv_args.append((direction, i, j))
        
        # Paraleliza o cálculo das derivadas
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(compute_derivative, deriv_args))
            for key, value in results:
                deriv_cache[key] = value
                # Simetria g_ij = g_ji
                deriv_cache[(key[0], key[2], key[1])] = value
        else:
            for args in deriv_args:
                key, value = compute_derivative(args)
                deriv_cache[key] = value
                deriv_cache[(key[0], key[2], key[1])] = value
        
        # Agora computa Christoffel usando o cache
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # Γ^k_ij = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
                    sum_term = 0.0
                    for l in range(dim):
                        dg_i = deriv_cache.get((i, j, l), 0.0)
                        dg_j = deriv_cache.get((j, i, l), 0.0)
                        dg_l = deriv_cache.get((l, i, j), 0.0)
                        sum_term += g_inv[k, l] * (dg_i + dg_j - dg_l)
                    gamma[k, i, j] = 0.5 * sum_term
                    
        return gamma

    def christoffel_at_active(self, x: np.ndarray, active_dims: int, eps: float = 1e-4) -> np.ndarray:
        """
        Calcula símbolos de Christoffel apenas para as primeiras `active_dims`.
        Otimizado com broadcasting e einsum.
        """
        dim = len(x)
        ad = min(active_dims, dim)
        
        g = self.metric_at(x)
        
        try:
            # Precisamos do inverso. Se a métrica for muito estável/diagonal, 
            # g_inv[0:ad, 0:ad] seria o inverso de g[0:ad, 0:ad], mas aqui usamos aproximação.
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            return np.zeros((ad, ad, ad))
            
        g_inv_sub = g_inv[:ad, :ad] # [k, l]
        
        # Derivadas parciais d g_... / d x_p
        # p corre 0..ad
        # matriz g completa é [dim, dim], mas truncamos para [ad, ad] para o cálculo
        
        # Vamos calcular dg para todos os p de uma vez?
        # metric_at é vetorizada sobre Deformations, mas não sobre pontos x múltiplos de forma trivial sem mudar a assinatura.
        # Loop sobre p (dimensões ativas) é aceitável (ad ~ 20-30), o custo maior era o loop interno i,j,l.
        
        dg_tensor = np.zeros((ad, ad, ad)) # [p, i, j] -> d g_ij / d x_p
        
        for p in range(ad):
            x_plus = x.copy(); x_plus[p] += eps
            x_minus = x.copy(); x_minus[p] -= eps
            gp = self.metric_at(x_plus)
            gm = self.metric_at(x_minus)
            
            # Truncamos gp e gm para [ad, ad]
            diff = (gp[:ad, :ad] - gm[:ad, :ad]) / (2 * eps)
            dg_tensor[p] = diff
            
        # Agora temos dg[p, i, j] = ∂_p g_ij
        # Formula: Γ^k_ij = 0.5 * g^kl * (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
        # 
        # Indices:
        # k -> output 0
        # i -> output 1
        # j -> output 2
        # l -> summation
        
        # Term 1: ∂_i g_jl -> dg[i, j, l]
        # Term 2: ∂_j g_il -> dg[j, i, l] = dg[j, l, i] (simetria de g_il) -> mas no tensor dg é p, row, col.
        #                  -> dg tensor é [p, row, col]. 
        #                  -> ∂_j g_il é derivada em j da entrada il. -> dg[j, i, l]
        # Term 3: ∂_l g_ij -> dg[l, i, j]
        
        # Vamos construir o tensor "S" (soma dos termos parentesis)
        # S[i, j, l] = dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
        
        # dg_tensor é [p, r, c].
        # Term1: i é index p, j é r, l é c -> dg_tensor
        # Term2: p=j, r=i, c=l -> transpose(0, 1) do dg_tensor?
        #        dg[j, i, l]. dg_tensor[j, i, l]. 
        #        Precisamos alinhar eixos.
        #        dg_tensor[a, b, c] -> derivada em a de g_bc.
        #        Queremos tensor [i, j, l].
        #        T1: dg[i, j, l] -> dg_tensor[i, j, l] (Elementwise)
        #        T2: dg[j, i, l] -> dg_tensor[j, i, l] (Permute 0,1 dos indices i,j)
        #        T3: dg[l, i, j] -> dg_tensor[l, i, j] (Permute 0,2 dos indices i,l ?)
        
        T1 = dg_tensor # [i, j, l]
        T2 = np.transpose(dg_tensor, (1, 0, 2)) # [j, i, l] -> swap p and r. p=j, r=i, c=l. Correct.
        T3 = np.transpose(dg_tensor, (2, 1, 0)) # [l, r, p] -> wait.
             # dg[l, i, j]. p=l, r=i, c=j.
             # dg_tensor original: [p, r, c].
             # Queremos mapear [i, j, l] para [p, r, c] onde p=l, r=i, c=j.
             # Entao a permutaçao de (0,1,2) [p,r,c] que leva a (l, i, j) ?
             # Não.
             # T3[i, j, l] deve vir de dg_tensor[l, i, j].
             # dg_tensor[l, i, j] tem índices (l, i, j).
             # Se eu quero um array T3 onde T3[i, j, l] == dg_tensor[l, i, j],
             # T3 é transposta de dg_tensor?
             # dg_tensor[A, B, C]. T3[B, C, A] = dg_tensor[A, B, C]. 
             # Sim, T3[i, j, l] com i=B, j=C, l=A.
             # Então transpose (1, 2, 0).
             
        T3 = np.transpose(dg_tensor, (1, 2, 0)) # [r, c, p] -> [i, j, l].
             # Check: T3[i, j, l] (indices 0,1,2 na nova matriz)
             # map to old: 0->1(r=i), 1->2(c=j), 2->0(p=l).
             # dg_tensor[l, i, j]. Correct.
             
        S = T1 + T2 - T3 # [i, j, l]
        
        # Agora contrair com g_inv[k, l]
        # Gamma[k, i, j] = 0.5 * sum_l (g_inv[k, l] * S[i, j, l])
        # Einsum: kl, ijl -> kij
        
        gamma = 0.5 * np.einsum('kl,ijl->kij', g_inv_sub, S)
        
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
