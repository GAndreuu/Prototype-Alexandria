"""
GeodesicFlow: Propagação por Caminhos Geodésicos
================================================

Geodésicas são os "caminhos mais curtos" em espaço curvo.
Quando a métrica é deformada, os caminhos retos Euclidianos
deixam de ser os mais curtos - a geodésica "curva" ao redor
das deformações.

Isso implementa como ativação se propaga: não em linha reta,
mas seguindo a geometria do espaço de conhecimento.

A equação geodésica é:
    d²x^k/dt² + Γ^k_ij (dx^i/dt)(dx^j/dt) = 0
    
Onde Γ são os símbolos de Christoffel derivados da métrica.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from scipy.integrate import solve_ivp

from .manifold import DynamicManifold, ManifoldPoint
from .metric import RiemannianMetric


@dataclass
class GeodesicConfig:
    """Configuração do fluxo geodésico."""
    max_steps: int = 100               # Máximo de passos por geodésica
    step_size: float = 0.01            # Tamanho do passo de integração
    tolerance: float = 1e-6            # Tolerância para convergência
    min_velocity: float = 0.001        # Velocidade mínima (para de propagar)
    damping: float = 0.01              # Amortecimento da velocidade
    use_scipy_integrator: bool = False # Usar integrador scipy (mais preciso, mais lento)


@dataclass
class GeodesicPath:
    """Um caminho geodésico."""
    points: np.ndarray           # [n_steps, dim] - pontos ao longo do caminho
    velocities: np.ndarray       # [n_steps, dim] - velocidades em cada ponto
    parameter: np.ndarray        # [n_steps] - parâmetro t ao longo do caminho
    length: float                # Comprimento total do caminho
    converged: bool             # Se atingiu destino/estabilizou
    
    @property
    def start(self) -> np.ndarray:
        return self.points[0]
    
    @property
    def end(self) -> np.ndarray:
        return self.points[-1]
    
    @property
    def n_steps(self) -> int:
        return len(self.points)


class GeodesicFlow:
    """
    Motor de propagação geodésica.
    
    Este componente é responsável por propagar ativação no espaço
    seguindo a geometria natural (geodésicas) em vez de caminhos
    Euclidianos.
    
    Attributes:
        manifold: Variedade base
        metric: Métrica Riemanniana
        config: Configuração
    """
    
    def __init__(self, 
                 manifold: DynamicManifold, 
                 metric: RiemannianMetric,
                 config: Optional[GeodesicConfig] = None):
        self.manifold = manifold
        self.metric = metric
        self.config = config or GeodesicConfig()
        
    # =========================================================================
    # GEODÉSICA ÚNICA
    # =========================================================================
    
    def compute_geodesic(self, 
                         start: np.ndarray, 
                         initial_velocity: np.ndarray,
                         max_steps: Optional[int] = None) -> GeodesicPath:
        """
        Computa uma geodésica a partir de ponto inicial com velocidade dada.
        
        A geodésica é a curva que satisfaz:
            d²x/dt² = -Γ^k_ij v^i v^j
            
        Args:
            start: Ponto inicial [dim]
            initial_velocity: Velocidade inicial [dim]
            max_steps: Máximo de passos (override config)
            
        Returns:
            GeodesicPath com a trajetória
        """
        max_steps = max_steps or self.config.max_steps
        
        if self.config.use_scipy_integrator:
            return self._geodesic_scipy(start, initial_velocity, max_steps)
        else:
            return self._geodesic_euler(start, initial_velocity, max_steps)
    
    def _geodesic_euler(self, 
                        start: np.ndarray, 
                        initial_velocity: np.ndarray,
                        max_steps: int) -> GeodesicPath:
        """
        Integração por Euler modificado (mais rápido, menos preciso).
        """
        dim = len(start)
        dt = self.config.step_size
        
        # Arrays para armazenar caminho
        points = [start.copy()]
        velocities = [initial_velocity.copy()]
        parameters = [0.0]
        
        x = start.copy()
        v = initial_velocity.copy()
        t = 0.0
        total_length = 0.0
        converged = False
        
        for step in range(max_steps):
            # Símbolos de Christoffel no ponto atual
            gamma = self.metric.christoffel_at(x)
            
            # Aceleração geodésica: a^k = -Γ^k_ij v^i v^j
            acceleration = np.zeros(dim)
            for k in range(dim):
                for i in range(dim):
                    for j in range(dim):
                        acceleration[k] -= gamma[k, i, j] * v[i] * v[j]
            
            # Atualiza velocidade (com damping opcional)
            v = v + acceleration * dt
            v = v * (1 - self.config.damping)
            
            # Atualiza posição
            x = x + v * dt
            
            # Comprimento do passo (usando métrica)
            g = self.metric.metric_at(x)
            ds = np.sqrt(np.abs(np.dot(v * dt, np.dot(g, v * dt))))
            total_length += ds
            
            # Armazena
            t += dt
            points.append(x.copy())
            velocities.append(v.copy())
            parameters.append(t)
            
            # Verifica convergência
            speed = np.linalg.norm(v)
            if speed < self.config.min_velocity:
                converged = True
                break
                
        return GeodesicPath(
            points=np.array(points),
            velocities=np.array(velocities),
            parameter=np.array(parameters),
            length=total_length,
            converged=converged
        )
    
    def _geodesic_scipy(self, 
                        start: np.ndarray, 
                        initial_velocity: np.ndarray,
                        max_steps: int) -> GeodesicPath:
        """
        Integração via scipy.integrate (mais preciso, mais lento).
        """
        dim = len(start)
        
        def geodesic_ode(t, state):
            """
            ODE para geodésica.
            State = [x, v] (posição e velocidade concatenados)
            """
            x = state[:dim]
            v = state[dim:]
            
            # Christoffel
            gamma = self.metric.christoffel_at(x)
            
            # dx/dt = v
            dx_dt = v
            
            # dv/dt = -Γ^k_ij v^i v^j
            dv_dt = np.zeros(dim)
            for k in range(dim):
                for i in range(dim):
                    for j in range(dim):
                        dv_dt[k] -= gamma[k, i, j] * v[i] * v[j]
                        
            return np.concatenate([dx_dt, dv_dt])
        
        # Condição inicial
        y0 = np.concatenate([start, initial_velocity])
        
        # Integra
        t_span = (0, max_steps * self.config.step_size)
        t_eval = np.linspace(*t_span, max_steps)
        
        try:
            sol = solve_ivp(
                geodesic_ode, 
                t_span, 
                y0, 
                t_eval=t_eval,
                method='RK45',
                rtol=self.config.tolerance
            )
            
            points = sol.y[:dim].T
            velocities = sol.y[dim:].T
            parameters = sol.t
            
            # Calcula comprimento
            total_length = 0.0
            for i in range(1, len(points)):
                dx = points[i] - points[i-1]
                g = self.metric.metric_at(points[i])
                ds = np.sqrt(np.abs(np.dot(dx, np.dot(g, dx))))
                total_length += ds
                
            converged = np.linalg.norm(velocities[-1]) < self.config.min_velocity
            
        except Exception as e:
            # Fallback para Euler se scipy falhar
            return self._geodesic_euler(start, initial_velocity, max_steps)
            
        return GeodesicPath(
            points=points,
            velocities=velocities,
            parameter=parameters,
            length=total_length,
            converged=converged
        )
    
    # =========================================================================
    # GEODÉSICA ENTRE DOIS PONTOS
    # =========================================================================
    
    def shortest_path(self, 
                      start: np.ndarray, 
                      end: np.ndarray,
                      max_iterations: int = 10) -> GeodesicPath:
        """
        Encontra geodésica conectando dois pontos.
        
        Isso é um problema de valor de contorno, mais difícil que
        o problema de valor inicial acima.
        
        Usa shooting method: ajusta velocidade inicial até acertar destino.
        
        Args:
            start: Ponto inicial [dim]
            end: Ponto final [dim]
            max_iterations: Iterações do shooting method
            
        Returns:
            GeodesicPath aproximada conectando os pontos
            
        TODO:
            - Implementar shooting method real
            - Ou usar relaxation method
        """
        # Por agora, aproximação simples: 
        # velocidade inicial na direção do destino
        direction = end - start
        distance = np.linalg.norm(direction)
        
        if distance < 1e-8:
            # Pontos coincidentes
            return GeodesicPath(
                points=np.array([start]),
                velocities=np.array([np.zeros_like(start)]),
                parameter=np.array([0.0]),
                length=0.0,
                converged=True
            )
            
        # Velocidade inicial: direção normalizada
        initial_velocity = direction / distance
        
        # Escala velocidade para chegar aproximadamente no destino
        # Isso é aproximado - geodésica real pode ter comprimento diferente
        estimated_time = distance / np.linalg.norm(initial_velocity)
        
        # Propaga
        path = self.compute_geodesic(
            start, 
            initial_velocity, 
            max_steps=int(estimated_time / self.config.step_size) + 10
        )
        
        return path
    
    # =========================================================================
    # PROPAGAÇÃO DE ATIVAÇÃO
    # =========================================================================
    
    def propagate_activation(self, 
                             source_point: ManifoldPoint,
                             steps: int = 3,
                             decay: float = 0.5) -> List[Tuple[str, float]]:
        """
        Propaga ativação de um ponto para vizinhos via geodésicas.
        
        Esta é a versão geodésica da propagação Hebbiana do mycelial.
        Em vez de seguir conexões do grafo, segue a geometria do espaço.
        
        Args:
            source_point: Ponto fonte da ativação
            steps: Número de passos de propagação
            decay: Fator de decaimento por passo
            
        Returns:
            Lista de (point_id, activation) para pontos atingidos
        """
        activations = {}
        
        # Encontra vizinhos do ponto fonte
        neighbors = self.manifold.get_neighbors(source_point)
        
        current_activation = source_point.activation
        frontier = [(source_point, current_activation)]
        visited = set()
        
        for step in range(steps):
            next_frontier = []
            
            for point, activation in frontier:
                if id(point) in visited:
                    continue
                visited.add(id(point))
                
                # Para cada vizinho, computa geodésica
                for neighbor_id, euclidean_dist in self.manifold.get_neighbors(point):
                    neighbor = self.manifold.get_point(neighbor_id)
                    if neighbor is None:
                        continue
                        
                    # Geodésica para o vizinho
                    path = self.shortest_path(point.coordinates, neighbor.coordinates)
                    
                    # Ativação decai com comprimento geodésico
                    geodesic_dist = path.length
                    propagated_activation = activation * np.exp(-geodesic_dist) * decay
                    
                    if propagated_activation > 0.01:  # Threshold
                        # Acumula ativação
                        if neighbor_id in activations:
                            activations[neighbor_id] = max(activations[neighbor_id], 
                                                          propagated_activation)
                        else:
                            activations[neighbor_id] = propagated_activation
                            
                        next_frontier.append((neighbor, propagated_activation))
                        
            frontier = next_frontier
            current_activation *= decay
            
        return list(activations.items())
    
    def flow_from_point(self, 
                        start: np.ndarray, 
                        n_directions: int = 8) -> List[GeodesicPath]:
        """
        Dispara geodésicas em múltiplas direções a partir de um ponto.
        
        Útil para visualizar como o espaço "se abre" a partir de um ponto.
        
        Args:
            start: Ponto inicial [dim]
            n_directions: Número de direções a explorar
            
        Returns:
            Lista de GeodesicPaths em diferentes direções
        """
        paths = []
        dim = len(start)
        
        # Gera direções (uniformes em esfera)
        # Para alta dimensão, amostra aleatória
        if dim > 10:
            directions = np.random.randn(n_directions, dim)
        else:
            # Para baixa dimensão, pode ser mais sistemático
            directions = np.random.randn(n_directions, dim)
            
        # Normaliza
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        for direction in directions:
            path = self.compute_geodesic(start, direction)
            paths.append(path)
            
        return paths
    
    # =========================================================================
    # CAMPO DE FLUXO
    # =========================================================================
    
    def flow_field(self, 
                   grid_points: np.ndarray,
                   direction_field: np.ndarray) -> List[GeodesicPath]:
        """
        Computa geodésicas partindo de múltiplos pontos.
        
        Args:
            grid_points: [n_points, dim] pontos iniciais
            direction_field: [n_points, dim] direções iniciais
            
        Returns:
            Lista de GeodesicPaths
        """
        paths = []
        
        for point, direction in zip(grid_points, direction_field):
            # Normaliza direção
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                path = self.compute_geodesic(point, direction)
                paths.append(path)
                
        return paths
    
    def streamlines(self, 
                    start_points: np.ndarray,
                    vector_field: callable,
                    max_length: float = 10.0) -> List[GeodesicPath]:
        """
        Computa streamlines seguindo um campo vetorial.
        
        Diferente de geodésicas puras, streamlines seguem
        um campo vetorial dado (ex: -∇F do FreeEnergyField).
        
        Args:
            start_points: [n_points, dim] pontos iniciais
            vector_field: função point -> velocity
            max_length: comprimento máximo de cada streamline
            
        Returns:
            Lista de paths (não são geodésicas, mas streamlines)
        """
        paths = []
        
        for start in start_points:
            points = [start.copy()]
            velocities = []
            length = 0.0
            
            current = start.copy()
            
            for _ in range(self.config.max_steps):
                # Velocidade do campo
                v = vector_field(current)
                velocities.append(v.copy())
                
                norm_v = np.linalg.norm(v)
                if norm_v < self.config.min_velocity:
                    break
                    
                # Passo
                dt = self.config.step_size
                current = current + v * dt
                
                # Comprimento
                g = self.metric.metric_at(current)
                ds = np.sqrt(np.abs(np.dot(v * dt, np.dot(g, v * dt))))
                length += ds
                
                points.append(current.copy())
                
                if length > max_length:
                    break
                    
            velocities.append(np.zeros_like(start))  # Final
            
            path = GeodesicPath(
                points=np.array(points),
                velocities=np.array(velocities),
                parameter=np.linspace(0, length, len(points)),
                length=length,
                converged=(length < max_length)
            )
            paths.append(path)
            
        return paths
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def geodesic_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Computa distância geodésica entre dois pontos.
        
        Esta é a "distância real" no espaço curvo.
        """
        path = self.shortest_path(p1, p2)
        return path.length
    
    def geodesic_midpoint(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Encontra ponto médio geodésico entre dois pontos.
        
        O ponto médio Euclidiano não é necessariamente o meio
        da geodésica em espaço curvo.
        """
        path = self.shortest_path(p1, p2)
        
        # Encontra ponto no meio do caminho (por comprimento)
        mid_idx = len(path.points) // 2
        return path.points[mid_idx]
    
    def parallel_transport(self, 
                          vector: np.ndarray, 
                          along_path: GeodesicPath) -> np.ndarray:
        """
        Transporta paralelamente um vetor ao longo de uma geodésica.
        
        Isso preserva o "ângulo" do vetor relativo à geodésica
        enquanto move pelo espaço curvo.
        
        TODO: Implementar via equação de transporte paralelo
        """
        # Por agora, retorna o vetor original (aproximação plana)
        return vector.copy()
    
    def stats(self) -> Dict:
        """Estatísticas do motor geodésico."""
        return {
            'config': self.config.__dict__,
            'manifold_dim': self.manifold.current_dim,
            'num_points': len(self.manifold.points)
        }
