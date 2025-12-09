"""
FreeEnergyField: Campo F(x,t) sobre a Variedade
===============================================

O campo de energia livre é a "paisagem" que determina a dinâmica do sistema.
Pontos de baixa energia livre são atratores - estados preferidos.
Gradientes indicam direções de "descida" natural.

F(x) = E(x) - T·S(x)

Onde:
- E(x) = energia interna (surpresa/prediction error)
- S(x) = entropia (incerteza)
- T = temperatura (exploration vs exploitation)

Conexão com VariationalFreeEnergy existente:
    O módulo free_energy.py já implementa F para estados discretos.
    Este módulo estende para campo contínuo sobre a variedade.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from .manifold import DynamicManifold, ManifoldPoint
from .metric import RiemannianMetric


@dataclass
class FieldConfig:
    """Configuração do campo de energia livre."""
    temperature: float = 1.0           # T na fórmula F = E - TS
    energy_scale: float = 1.0          # Escala da energia
    entropy_scale: float = 1.0         # Escala da entropia
    smoothing_sigma: float = 0.1       # Suavização do campo
    grid_resolution: int = 32          # Resolução para visualização
    well_threshold: float = -0.5       # Threshold para detectar poços


@dataclass
class FieldState:
    """Estado do campo em um instante."""
    energy_field: np.ndarray           # E(x) em grid
    entropy_field: np.ndarray          # S(x) em grid
    free_energy_field: np.ndarray      # F(x) = E - TS
    curvature_field: np.ndarray        # Curvatura da métrica
    gradient_field: np.ndarray         # ∇F em cada ponto
    attractors: List[np.ndarray]       # Mínimos locais de F
    grid_points: np.ndarray            # Pontos do grid [n, dim]
    
    @property
    def mean_free_energy(self) -> float:
        return np.mean(self.free_energy_field)
    
    @property
    def num_attractors(self) -> int:
        return len(self.attractors)


class FreeEnergyField:
    """
    Campo de Energia Livre sobre a variedade.
    
    Este é o coração do sistema - a paisagem que determina onde
    o conhecimento "quer" ir, quais estados são preferidos,
    e como a dinâmica evolui.
    
    Attributes:
        manifold: Variedade base
        metric: Métrica Riemanniana
        config: Configuração
        vfe: VariationalFreeEnergy existente (opcional, para reusar)
    """
    
    def __init__(self, 
                 manifold: DynamicManifold, 
                 metric: RiemannianMetric,
                 config: Optional[FieldConfig] = None,
                 variational_fe: Optional[any] = None):
        self.manifold = manifold
        self.metric = metric
        self.config = config or FieldConfig()
        self.vfe = variational_fe  # Conexão com módulo existente
        
        # Cache
        self._last_state: Optional[FieldState] = None
        self._cache_valid = False
        
    # =========================================================================
    # COMPONENTES DO CAMPO
    # =========================================================================
    
    def energy_at(self, point: np.ndarray) -> float:
        """
        Computa energia interna E(x) em um ponto.
        
        E(x) representa "surpresa" ou "prediction error".
        Alto E = estado inesperado/improvável.
        Baixo E = estado esperado/provável.
        
        Args:
            point: Coordenadas [dim]
            
        Returns:
            Energia no ponto
            
        TODO:
            - Integrar com IsomorphicPredictiveCoding para prediction error real
            - Usar VQ-VAE reconstruction error como proxy
        """
        # Energia base: distância aos pontos ativos
        active_points = self.manifold.get_active_points()
        
        if not active_points:
            # Sem ativação, energia uniforme
            return 0.0
            
        # Energia diminui perto de pontos ativos
        min_dist = float('inf')
        weighted_energy = 0.0
        total_weight = 0.0
        
        for pid, p in active_points:
            dist = np.linalg.norm(point - p.coordinates)
            weight = p.activation
            
            # Energia é baixa perto de pontos ativos
            # Usa métrica Riemanniana para distância "real"
            if dist < 0.001:
                return -p.activation * self.config.energy_scale
                
            contribution = weight / (dist + 0.1)
            weighted_energy += contribution
            total_weight += weight
            
        # Inverte: mais contribuição = menor energia
        if total_weight > 0:
            energy = -weighted_energy / total_weight
        else:
            energy = 0.0
            
        return energy * self.config.energy_scale
    
    def entropy_at(self, point: np.ndarray) -> float:
        """
        Computa entropia S(x) em um ponto.
        
        S(x) representa incerteza sobre transições a partir desse ponto.
        Alto S = muitas possibilidades.
        Baixo S = poucas/determinísticas possibilidades.
        
        Args:
            point: Coordenadas [dim]
            
        Returns:
            Entropia no ponto
            
        TODO:
            - Calcular baseado em distribuição de vizinhos
            - Integrar com beliefs do ActiveInference
        """
        # Entropia baseada em densidade local
        neighbors = self.manifold.get_neighbors(
            ManifoldPoint(coordinates=point, discrete_codes=np.zeros(4, dtype=np.int32)),
            k=min(16, len(self.manifold.points))
        )
        
        if not neighbors:
            # Ponto isolado = alta entropia
            return 1.0 * self.config.entropy_scale
            
        # Distâncias aos vizinhos
        distances = [d for _, d in neighbors]
        
        if len(distances) < 2:
            return 1.0 * self.config.entropy_scale
            
        # Entropia ~ variância das distâncias
        # Vizinhos uniformemente distribuídos = alta entropia
        # Vizinhos em cluster = baixa entropia
        mean_dist = np.mean(distances)
        var_dist = np.var(distances)
        
        # Normaliza
        entropy = var_dist / (mean_dist + 0.1)
        
        return entropy * self.config.entropy_scale
    
    def free_energy_at(self, point: np.ndarray) -> float:
        """
        Computa energia livre F(x) = E(x) - T·S(x) em um ponto.
        
        Args:
            point: Coordenadas [dim]
            
        Returns:
            Energia livre no ponto
            
        Interpretação:
            - Baixo F = estado preferido (baixa energia, ou alta entropia com alta T)
            - Alto F = estado evitado
            - Gradiente de F indica direção de "descida"
        """
        E = self.energy_at(point)
        S = self.entropy_at(point)
        T = self.config.temperature
        
        F = E - T * S
        
        return F
    
    # =========================================================================
    # CAMPO COMPLETO
    # =========================================================================
    
    def compute_field(self, grid_points: Optional[np.ndarray] = None) -> FieldState:
        """
        Computa campo de energia livre em grid de pontos.
        
        Args:
            grid_points: [n_points, dim] ou None para usar pontos da variedade
            
        Returns:
            FieldState com todos os campos computados
        """
        # Define grid
        if grid_points is None:
            if len(self.manifold.points) == 0:
                # Sem pontos, retorna estado vazio
                return FieldState(
                    energy_field=np.array([]),
                    entropy_field=np.array([]),
                    free_energy_field=np.array([]),
                    curvature_field=np.array([]),
                    gradient_field=np.array([]),
                    attractors=[],
                    grid_points=np.array([])
                )
            grid_points = self.manifold.get_coordinates_matrix()
            
        n_points = len(grid_points)
        
        # Computa campos
        energy = np.array([self.energy_at(p) for p in grid_points])
        entropy = np.array([self.entropy_at(p) for p in grid_points])
        free_energy = energy - self.config.temperature * entropy
        
        # Curvatura da métrica
        curvature = self.metric.curvature_tensor(grid_points)
        
        # Gradientes
        gradient = np.array([self.gradient_at(p) for p in grid_points])
        
        # Encontra atratores (mínimos locais)
        attractors = self._find_attractors(grid_points, free_energy)
        
        state = FieldState(
            energy_field=energy,
            entropy_field=entropy,
            free_energy_field=free_energy,
            curvature_field=curvature,
            gradient_field=gradient,
            attractors=attractors,
            grid_points=grid_points
        )
        
        self._last_state = state
        self._cache_valid = True
        
        return state
    
    def gradient_at(self, point: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        Computa gradiente ∇F em um ponto.
        
        O gradiente indica a direção de maior aumento de F.
        O sistema "desce" na direção oposta (-∇F).
        
        Args:
            point: Coordenadas [dim]
            epsilon: Step para diferenças finitas
            
        Returns:
            Vetor gradiente [dim]
        """
        dim = len(point)
        grad = np.zeros(dim)
        
        F_center = self.free_energy_at(point)
        
        # Computa derivadas parciais
        # Para eficiência, só primeiras dimensões em alta dimensão
        max_dims = min(dim, 20)
        
        for i in range(max_dims):
            point_plus = point.copy()
            point_plus[i] += epsilon
            F_plus = self.free_energy_at(point_plus)
            grad[i] = (F_plus - F_center) / epsilon
            
        return grad
    
    def _find_attractors(self, grid_points: np.ndarray, 
                         free_energy: np.ndarray) -> List[np.ndarray]:
        """
        Encontra mínimos locais de F (atratores).
        
        TODO:
            - Algoritmo mais sofisticado (watershed, gradient descent)
            - Merge de atratores muito próximos
        """
        attractors = []
        threshold = self.config.well_threshold
        
        # Para cada ponto, verifica se é mínimo local
        for i, (point, F) in enumerate(zip(grid_points, free_energy)):
            if F > threshold:
                continue
                
            # Verifica se é menor que vizinhos
            is_minimum = True
            
            for j, other_F in enumerate(free_energy):
                if i == j:
                    continue
                dist = np.linalg.norm(grid_points[j] - point)
                if dist < 0.3 and other_F < F:  # Vizinho com F menor
                    is_minimum = False
                    break
                    
            if is_minimum:
                attractors.append(point.copy())
                
        return attractors
    
    # =========================================================================
    # DINÂMICA
    # =========================================================================
    
    def descend(self, point: np.ndarray, step_size: float = 0.01, 
                steps: int = 1) -> np.ndarray:
        """
        Desce no gradiente de F a partir de um ponto.
        
        Simula a evolução natural do sistema em direção
        a estados de menor energia livre.
        
        Args:
            point: Ponto inicial [dim]
            step_size: Tamanho do passo
            steps: Número de passos
            
        Returns:
            Ponto final após descida
        """
        current = point.copy()
        
        for _ in range(steps):
            grad = self.gradient_at(current)
            
            # Usa métrica para "corrigir" a direção
            # Em espaço curvo, -∇F não é necessariamente o melhor caminho
            g_inv = self.metric.metric_tensor_inverse(current)
            natural_grad = np.dot(g_inv, grad)
            
            # Normaliza para não pular muito
            norm = np.linalg.norm(natural_grad)
            if norm > 0.01:
                natural_grad = natural_grad / norm
                
            # Passo na direção de menor F
            current = current - step_size * natural_grad
            
        return current
    
    def flow_field(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Computa campo de fluxo (velocidades) em grid.
        
        O fluxo indica para onde pontos "querem" ir.
        
        Args:
            grid_points: [n_points, dim]
            
        Returns:
            [n_points, dim] com vetores de velocidade
        """
        velocities = []
        
        for point in grid_points:
            grad = self.gradient_at(point)
            # Velocidade = -∇F (desce gradiente)
            velocity = -grad
            velocities.append(velocity)
            
        return np.array(velocities)
    
    # =========================================================================
    # TEMPERATURA
    # =========================================================================
    
    def set_temperature(self, T: float):
        """
        Ajusta temperatura do sistema.
        
        T alto: entropia domina, sistema explora
        T baixo: energia domina, sistema explota
        
        Isso é útil para simulated annealing.
        """
        self.config.temperature = T
        self._cache_valid = False
        
    def anneal(self, schedule: Callable[[int], float], steps: int):
        """
        Aplica schedule de annealing.
        
        Args:
            schedule: Função step -> temperatura
            steps: Número de passos
            
        Exemplo:
            field.anneal(lambda t: 1.0 - t/100, steps=100)
        """
        for t in range(steps):
            T = schedule(t)
            self.set_temperature(T)
            # TODO: Evoluir sistema a cada passo
            
    # =========================================================================
    # CONEXÃO COM MÓDULOS EXISTENTES
    # =========================================================================
    
    def sync_with_variational_fe(self):
        """
        Sincroniza com módulo VariationalFreeEnergy existente.
        
        TODO:
            - Importar beliefs do módulo
            - Usar accuracy/complexity para E e S
        """
        if self.vfe is None:
            return
            
        # TODO: Implementar integração
        # self.vfe.get_beliefs() -> ajustar campos
        pass
    
    def update_from_prediction_error(self, prediction_errors: Dict[str, float]):
        """
        Atualiza campo de energia baseado em erros de predição.
        
        Integração com IsomorphicPredictiveCoding.
        
        Args:
            prediction_errors: {point_id: error} 
        """
        for point_id, error in prediction_errors.items():
            point = self.manifold.get_point(point_id)
            if point:
                # Erro alto = energia alta
                point.metadata['prediction_error'] = error
                
        self._cache_valid = False
        
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def get_state(self) -> Optional[FieldState]:
        """Retorna último estado computado."""
        if not self._cache_valid:
            return self.compute_field()
        return self._last_state
    
    def stats(self) -> Dict:
        """Estatísticas do campo."""
        state = self.get_state()
        if state is None or len(state.free_energy_field) == 0:
            return {
                'mean_F': 0.0,
                'min_F': 0.0,
                'max_F': 0.0,
                'num_attractors': 0,
                'temperature': self.config.temperature
            }
            
        return {
            'mean_F': float(np.mean(state.free_energy_field)),
            'min_F': float(np.min(state.free_energy_field)),
            'max_F': float(np.max(state.free_energy_field)),
            'std_F': float(np.std(state.free_energy_field)),
            'mean_E': float(np.mean(state.energy_field)),
            'mean_S': float(np.mean(state.entropy_field)),
            'num_attractors': state.num_attractors,
            'temperature': self.config.temperature
        }
