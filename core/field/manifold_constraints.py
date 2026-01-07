"""
Manifold Constraints: Safety Layer baseado em mHC (DeepSeek)
============================================================

Implementa projeções no Politopo de Birkhoff para garantir:
- Preservação de norma: ||H||_2 ≤ 1
- Combinações convexas: Σ w_i = 1, w_i ≥ 0
- Estabilidade em propagações profundas

Referência:
    mHC: Manifold-Constrained Hyper-Connections (DeepSeek-AI, 2025)
    arXiv:2512.24880v1

Autor: Alexandria System
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SINKHORN-KNOPP PROJECTION
# =============================================================================

def sinkhorn_knopp(
    log_weights: np.ndarray, 
    iterations: int = 20,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Projeta matriz no Politopo de Birkhoff via Sinkhorn-Knopp.
    
    O algoritmo converge para uma matriz duplamente estocástica onde:
    - Todas linhas somam 1
    - Todas colunas somam 1
    - Todos elementos ≥ 0
    
    Isso garante ||H||_2 ≤ 1 (não-expansivo).
    
    Args:
        log_weights: Matriz de log-pesos [n, n] ou vetor [n]
                     Aceita valores negativos (serão exponenciados)
        iterations: Número de iterações (paper usa 20)
        eps: Estabilidade numérica para divisão
    
    Returns:
        Matriz/vetor duplamente estocástico
        
    Referência:
        Eq. (8) e (9) do paper mHC
    """
    # Garante positividade via exp com estabilização
    log_weights = np.asarray(log_weights, dtype=np.float64)
    M = np.exp(log_weights - np.max(log_weights))  # Subtract max for numerical stability
    
    is_vector = M.ndim == 1
    if is_vector:
        M = M.reshape(1, -1)
    
    # Iterações alternadas de normalização (T_r e T_c do paper)
    for _ in range(iterations):
        # T_r: Normaliza linhas para somarem 1
        row_sums = M.sum(axis=-1, keepdims=True)
        M = M / (row_sums + eps)
        
        # T_c: Normaliza colunas para somarem 1
        col_sums = M.sum(axis=-2, keepdims=True)
        M = M / (col_sums + eps)
    
    return M.flatten() if is_vector else M


def normalize_weights_convex(
    weights: np.ndarray,
    use_sinkhorn: bool = True,
    iterations: int = 20
) -> np.ndarray:
    """
    Normaliza pesos para formar combinação convexa.
    
    Para vetores: Σ w_i = 1, w_i ≥ 0
    Para matrizes: Duplamente estocástica (via Sinkhorn)
    
    Args:
        weights: Pesos brutos (qualquer escala)
        use_sinkhorn: Se True, usa Sinkhorn para matrizes
        iterations: Iterações Sinkhorn se aplicável
    
    Returns:
        Pesos normalizados formando combinação convexa
    """
    weights = np.asarray(weights, dtype=np.float64)
    
    if weights.ndim == 2 and use_sinkhorn:
        # Matriz: usa Sinkhorn completo
        return sinkhorn_knopp(np.log(np.abs(weights) + 1e-8), iterations)
    else:
        # Vetor: normalização simplex simples
        w = np.abs(weights)
        total = np.sum(w)
        if total < 1e-8:
            # Evita divisão por zero - retorna uniforme
            return np.ones_like(w) / len(w)
        return w / total


# =============================================================================
# SPECTRAL NORM CONTROL
# =============================================================================

def spectral_norm(matrix: np.ndarray) -> float:
    """
    Calcula norma espectral (maior valor singular) de uma matriz.
    
    Args:
        matrix: Matriz [m, n]
    
    Returns:
        ||M||_2 (norma espectral)
    """
    if matrix.size == 0:
        return 0.0
    return np.linalg.svd(matrix, compute_uv=False)[0]


def spectral_norm_clip(
    matrix: np.ndarray,
    max_norm: float = 1.0
) -> np.ndarray:
    """
    Limita norma espectral da matriz para garantir não-expansividade.
    
    Se ||M||_2 > max_norm, escala a matriz para ter exatamente max_norm.
    Preserva a direção das transformações, apenas limita a magnitude.
    
    Args:
        matrix: Matriz a clipar
        max_norm: Norma máxima permitida (default: 1.0 para não-expansivo)
    
    Returns:
        Matriz com ||M||_2 ≤ max_norm
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    
    if matrix.size == 0:
        return matrix
        
    current_norm = spectral_norm(matrix)
    
    if current_norm > max_norm:
        # Escala proporcional
        scale = max_norm / current_norm
        return matrix * scale
    
    return matrix


def spectral_norm_clip_svd(
    matrix: np.ndarray,
    max_norm: float = 1.0
) -> np.ndarray:
    """
    Limita norma espectral via SVD truncado.
    
    Mais preciso que scaling simples - clippa valores singulares
    individualmente enquanto preserva a estrutura.
    
    Args:
        matrix: Matriz a clipar
        max_norm: Norma máxima por valor singular
    
    Returns:
        Matriz com todos valores singulares ≤ max_norm
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    
    if matrix.size == 0:
        return matrix
    
    try:
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        S_clipped = np.minimum(S, max_norm)
        return U @ np.diag(S_clipped) @ Vh
    except np.linalg.LinAlgError:
        # Fallback para scaling simples se SVD falhar
        return spectral_norm_clip(matrix, max_norm)


# =============================================================================
# JACOBIAN ANALYSIS
# =============================================================================

def jacobian_norm_check(
    jacobian: np.ndarray,
    threshold: float = 1.5
) -> Tuple[bool, float]:
    """
    Verifica se Jacobiano indica expansão perigosa.
    
    O Jacobiano de uma transformação x → f(x) indica quanto
    a transformação expande/contrai localmente.
    
    ||J|| > 1 indica expansão (potencialmente instável em iterações)
    ||J|| < 1 indica contração (estável)
    ||J|| = 1 indica preservação (isometria)
    
    Args:
        jacobian: Matriz Jacobiana [dim, dim]
        threshold: Limiar para considerar perigoso (default: 1.5)
    
    Returns:
        (is_safe, spectral_norm): Tupla com flag de segurança e norma
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    
    if jacobian.size == 0:
        return True, 0.0
    
    norm = spectral_norm(jacobian)
    is_safe = norm <= threshold
    
    if not is_safe:
        logger.warning(f"Jacobian norm {norm:.3f} exceeds threshold {threshold}")
    
    return is_safe, norm


def estimate_local_expansion(
    func,
    point: np.ndarray,
    epsilon: float = 1e-4
) -> float:
    """
    Estima expansão local de uma função via diferenças finitas.
    
    Args:
        func: Função f: R^n → R^n
        point: Ponto onde estimar
        epsilon: Perturbação para diferenças finitas
    
    Returns:
        Estimativa de ||J||_2 no ponto
    """
    point = np.asarray(point, dtype=np.float64)
    dim = len(point)
    
    # Constrói Jacobiano por diferenças finitas
    jacobian = np.zeros((dim, dim))
    f0 = func(point)
    
    for i in range(dim):
        point_plus = point.copy()
        point_plus[i] += epsilon
        f_plus = func(point_plus)
        jacobian[:, i] = (f_plus - f0) / epsilon
    
    return spectral_norm(jacobian)


# =============================================================================
# ENERGY PRESERVATION
# =============================================================================

def kinetic_energy(velocity: np.ndarray, metric: Optional[np.ndarray] = None) -> float:
    """
    Calcula energia cinética v^T G v.
    
    Args:
        velocity: Vetor velocidade [dim]
        metric: Tensor métrico [dim, dim] (default: identidade)
    
    Returns:
        Energia cinética (escalar ≥ 0)
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    
    if metric is None:
        return float(np.dot(velocity, velocity))
    
    return float(velocity @ metric @ velocity)


def cap_kinetic_energy(
    velocity: np.ndarray,
    current_energy: float,
    max_energy: float,
    metric: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Limita energia cinética renormalizando velocidade.
    
    Preserva direção do movimento, apenas limita magnitude.
    
    Args:
        velocity: Velocidade atual [dim]
        current_energy: Energia cinética atual
        max_energy: Energia máxima permitida
        metric: Tensor métrico (para cálculo correto em espaço curvo)
    
    Returns:
        Velocidade ajustada com energia ≤ max_energy
    """
    if current_energy <= max_energy or current_energy < 1e-12:
        return velocity
    
    # Fator de escala para atingir max_energy
    scale = np.sqrt(max_energy / current_energy)
    return velocity * scale


# =============================================================================
# CONVEX COMBINATION UTILITIES
# =============================================================================

def barycentric_coordinates(
    point: np.ndarray,
    anchors: np.ndarray,
    use_sinkhorn: bool = True
) -> np.ndarray:
    """
    Calcula coordenadas baricêntricas de um ponto em relação a âncoras.
    
    Usa distância inversa com normalização Sinkhorn para garantir
    que o ponto seja combinação convexa das âncoras.
    
    Args:
        point: Ponto a projetar [dim]
        anchors: Pontos âncora [n_anchors, dim]
        use_sinkhorn: Usar Sinkhorn para normalização
    
    Returns:
        Pesos baricêntricos [n_anchors] onde Σ w_i = 1, w_i ≥ 0
    """
    point = np.asarray(point, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    
    # Distâncias para cada âncora
    distances = np.linalg.norm(anchors - point, axis=1)
    
    # Afinidade: inversamente proporcional à distância
    # Usa -distância como log-afinidade
    log_affinities = -distances
    
    return normalize_weights_convex(np.exp(log_affinities), use_sinkhorn=False)


def interpolate_barycentic(
    weights: np.ndarray,
    values: np.ndarray
) -> np.ndarray:
    """
    Interpola valores usando pesos baricêntricos.
    
    Args:
        weights: Pesos [n] (devem somar 1)
        values: Valores a interpolar [n, dim]
    
    Returns:
        Valor interpolado [dim]
    """
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    
    return np.sum(values * weights[:, np.newaxis], axis=0)


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def check_doubly_stochastic(matrix: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Verifica se matriz é duplamente estocástica.
    
    Args:
        matrix: Matriz a verificar
        tol: Tolerância numérica
    
    Returns:
        True se duplamente estocástica
    """
    matrix = np.asarray(matrix)
    
    if matrix.ndim != 2:
        return False
    
    # Verifica não-negatividade
    if np.any(matrix < -tol):
        return False
    
    # Verifica soma de linhas
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False
    
    # Verifica soma de colunas
    col_sums = matrix.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=tol):
        return False
    
    return True


def check_convex_weights(weights: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Verifica se pesos formam combinação convexa válida.
    
    Args:
        weights: Vetor de pesos
        tol: Tolerância numérica
    
    Returns:
        True se soma = 1 e todos ≥ 0
    """
    weights = np.asarray(weights)
    
    # Verifica não-negatividade
    if np.any(weights < -tol):
        return False
    
    # Verifica soma = 1
    if not np.isclose(weights.sum(), 1.0, atol=tol):
        return False
    
    return True
