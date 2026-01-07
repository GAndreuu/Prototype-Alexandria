"""
Loop ↔ Compositional Integration
=================================

Conecta o Self-Feeding Loop ao raciocínio composicional, fechando
o ciclo de aprendizado autônomo.

Isso permite:
- Self-Feeding Loop gerar hipóteses como caminhos geodésicos
- Nemesis calcular EFE no manifold curvo
- Feedback propagar ao longo de estruturas geométricas
- Ciclo completo: Perceber → Raciocinar → Agir → Aprender

Teoria:
    Loop autônomo no manifold:
    
    1. PERCEBER: observation → embed → ManifoldPoint
    2. RACIOCINAR: point → geodesic_reasoning → hypothesis
    3. AGIR: hypothesis → action → new_state  
    4. APRENDER: (state, action, reward) → deform_metric
    
    A métrica evolui com aprendizado, tornando caminhos
    úteis mais "fáceis" de percorrer.

Uso:
    from loop_compositional_integration import LoopCompositionalIntegration
    
    lci = LoopCompositionalIntegration(bridge, compositional, loop)
    
    # Ciclo completo
    result = lci.autonomous_cycle(observation, goal)
    
    # Auto-feeding com geodésicas
    hypotheses = lci.generate_geodesic_hypotheses(gap)

Autor: G (Alexandria Project)
Versão: 1.0
Fase: 4.1 - Ciclo Fechado
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class LoopCompositionalConfig:
    """Configuração da integração Loop ↔ Compositional."""
    
    # Ciclo
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    energy_target: float = 0.1
    
    # Self-Feeding
    hypothesis_generation_interval: int = 10
    max_hypotheses_per_cycle: int = 5
    min_hypothesis_confidence: float = 0.3
    
    # Learning
    metric_learning_rate: float = 0.01
    reward_discount: float = 0.95
    exploration_rate: float = 0.1
    
    # Feedback
    feedback_propagation_depth: int = 3
    feedback_decay: float = 0.9
    
    # Checkpointing
    checkpoint_interval: int = 50
    save_trajectory: bool = True


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

class LoopPhase(Enum):
    """Fases do loop autônomo."""
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    LEARN = "learn"


@dataclass
class LoopState:
    """Estado do loop em um momento."""
    iteration: int
    phase: LoopPhase
    
    # Estado no manifold
    current_point: np.ndarray
    belief: np.ndarray
    
    # Métricas
    free_energy: float
    prediction_error: float
    cumulative_reward: float
    
    # Histórico curto
    recent_actions: List[Dict] = field(default_factory=list)
    recent_hypotheses: List[Dict] = field(default_factory=list)


@dataclass
class CycleResult:
    """Resultado de um ciclo completo."""
    # Trajeto
    trajectory: List[np.ndarray]
    actions_taken: List[Dict]
    
    # Métricas
    total_iterations: int
    final_free_energy: float
    energy_reduction: float
    hypotheses_generated: int
    hypotheses_validated: int
    
    # Aprendizado
    metric_deformations: int
    cumulative_reward: float
    
    # Tempo
    duration_seconds: float


@dataclass
class GeodesicFeedback:
    """Feedback propagado via geodésica."""
    source_point: np.ndarray
    affected_points: List[np.ndarray]
    feedback_values: List[float]
    propagation_path: np.ndarray
    total_impact: float


# =============================================================================
# INTEGRAÇÃO PRINCIPAL
# =============================================================================

class LoopCompositionalIntegration:
    """
    Integração do Self-Feeding Loop com raciocínio composicional.
    
    Fecha o ciclo autônomo onde o sistema:
    1. Percebe via embedding no manifold
    2. Raciocina via geodésicas composicionais
    3. Age com base em EFE geométrico
    4. Aprende deformando a métrica
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        self_feeding_loop=None,
        nemesis=None,
        abduction=None,
        config: Optional[LoopCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.self_feeding = self_feeding_loop
        self.nemesis = nemesis
        self.abduction = abduction
        self.config = config or LoopCompositionalConfig()
        
        # Estado
        self._current_state: Optional[LoopState] = None
        self._trajectory: List[np.ndarray] = []
        self._metric_updates: List[Dict] = []
        self._cumulative_reward = 0.0
        
        logger.info("LoopCompositionalIntegration initialized")
    
    # =========================================================================
    # CICLO AUTÔNOMO
    # =========================================================================
    
    def autonomous_cycle(
        self,
        initial_observation: np.ndarray,
        goal: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None
    ) -> CycleResult:
        """
        Executa ciclo autônomo completo até convergência.
        
        Args:
            initial_observation: Observação inicial
            goal: Goal opcional (se None, minimiza F)
            max_iterations: Limite de iterações
        
        Returns:
            CycleResult com métricas do ciclo
        """
        start_time = time.time()
        max_iter = max_iterations or self.config.max_iterations
        
        # Inicializar
        self._init_cycle(initial_observation, goal)
        initial_energy = self._current_state.free_energy
        
        hypotheses_generated = 0
        hypotheses_validated = 0
        actions_taken = []
        
        # Loop principal
        for i in range(max_iter):
            # Verificar convergência
            if self._check_convergence():
                logger.info(f"Converged at iteration {i}")
                break
            
            # Executar fases
            # 1. PERCEIVE
            self._perceive_phase()
            
            # 2. REASON
            hypothesis = self._reason_phase()
            if hypothesis is not None:
                hypotheses_generated += 1
            
            # 3. ACT
            action, reward = self._act_phase(hypothesis, goal)
            if action is not None:
                actions_taken.append(action)
            
            # 4. LEARN
            validated = self._learn_phase(action, reward)
            if validated:
                hypotheses_validated += 1
            
            # Checkpoint
            if i % self.config.checkpoint_interval == 0:
                self._checkpoint(i)
        
        duration = time.time() - start_time
        
        return CycleResult(
            trajectory=self._trajectory.copy(),
            actions_taken=actions_taken,
            total_iterations=i + 1,
            final_free_energy=self._current_state.free_energy,
            energy_reduction=initial_energy - self._current_state.free_energy,
            hypotheses_generated=hypotheses_generated,
            hypotheses_validated=hypotheses_validated,
            metric_deformations=len(self._metric_updates),
            cumulative_reward=self._cumulative_reward,
            duration_seconds=duration
        )
    
    def _init_cycle(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray]
    ):
        """Inicializa estado do ciclo."""
        # Projetar para manifold
        if hasattr(self.bridge, '_project_to_latent'):
            point = self.bridge._project_to_latent(observation)
        else:
            point = observation
        
        # Calcular F inicial
        F = self._compute_free_energy(point)
        
        # Criar estado
        self._current_state = LoopState(
            iteration=0,
            phase=LoopPhase.PERCEIVE,
            current_point=point,
            belief=point.copy(),
            free_energy=F,
            prediction_error=0.0,
            cumulative_reward=0.0,
            recent_actions=[],
            recent_hypotheses=[]
        )
        
        # Reset histórico
        self._trajectory = [point.copy()]
        self._metric_updates = []
        self._cumulative_reward = 0.0
    
    def _check_convergence(self) -> bool:
        """Verifica se ciclo convergiu."""
        if self._current_state is None:
            return False
        
        # Convergiu se F baixo
        if self._current_state.free_energy < self.config.energy_target:
            return True
        
        # Convergiu se não há mais mudança
        if len(self._trajectory) >= 3:
            recent = self._trajectory[-3:]
            deltas = [np.linalg.norm(recent[i+1] - recent[i]) for i in range(2)]
            if max(deltas) < self.config.convergence_threshold:
                return True
        
        return False
    
    # =========================================================================
    # FASES DO CICLO
    # =========================================================================
    
    def _perceive_phase(self):
        """
        Fase PERCEIVE: atualiza belief baseado na observação atual.
        """
        self._current_state.phase = LoopPhase.PERCEIVE
        
        point = self._current_state.current_point
        belief = self._current_state.belief
        
        # Prediction error
        error = np.linalg.norm(point - belief)
        self._current_state.prediction_error = error
        
        # Atualizar belief (média móvel)
        lr = 0.3
        self._current_state.belief = (1 - lr) * belief + lr * point
        
        # Atualizar F
        self._current_state.free_energy = self._compute_free_energy(point)
    
    def _reason_phase(self) -> Optional[Dict]:
        """
        Fase REASON: gera hipóteses via raciocínio composicional.
        """
        self._current_state.phase = LoopPhase.REASON
        
        # Verificar se deve gerar hipótese
        if self._current_state.iteration % self.config.hypothesis_generation_interval != 0:
            return None
        
        point = self._current_state.current_point
        
        # Encontrar gaps locais
        gaps = self._detect_local_gaps(point)
        
        if not gaps:
            return None
        
        # Gerar hipótese geodésica para gap mais promissor
        gap = gaps[0]
        hypothesis = self._generate_geodesic_hypothesis(gap)
        
        if hypothesis is not None:
            self._current_state.recent_hypotheses.append(hypothesis)
            # Manter só as últimas 5
            self._current_state.recent_hypotheses = self._current_state.recent_hypotheses[-5:]
        
        return hypothesis
    
    def _act_phase(
        self,
        hypothesis: Optional[Dict],
        goal: Optional[np.ndarray]
    ) -> Tuple[Optional[Dict], float]:
        """
        Fase ACT: executa ação baseada em hipótese ou exploração.
        """
        self._current_state.phase = LoopPhase.ACT
        
        point = self._current_state.current_point
        
        # Decidir ação
        if np.random.random() < self.config.exploration_rate:
            # Explorar
            action = self._explore_action(point)
        elif hypothesis is not None:
            # Seguir hipótese
            action = self._hypothesis_action(hypothesis)
        elif goal is not None:
            # Mover para goal
            action = self._goal_action(point, goal)
        else:
            # Minimizar F
            action = self._minimize_f_action(point)
        
        # Executar ação
        new_point = self._execute_action(point, action)
        
        # Calcular recompensa
        old_F = self._current_state.free_energy
        new_F = self._compute_free_energy(new_point)
        reward = old_F - new_F  # Recompensa por reduzir F
        
        # Atualizar estado
        self._current_state.current_point = new_point
        self._current_state.free_energy = new_F
        self._current_state.cumulative_reward += reward
        self._cumulative_reward += reward
        
        # Salvar trajeto
        if self.config.save_trajectory:
            self._trajectory.append(new_point.copy())
        
        # Registrar ação
        self._current_state.recent_actions.append(action)
        self._current_state.recent_actions = self._current_state.recent_actions[-5:]
        
        return action, reward
    
    def _learn_phase(
        self,
        action: Optional[Dict],
        reward: float
    ) -> bool:
        """
        Fase LEARN: atualiza métrica e valida hipóteses.
        """
        self._current_state.phase = LoopPhase.LEARN
        self._current_state.iteration += 1
        
        validated = False
        
        # Atualizar métrica se recompensa positiva
        if reward > 0 and action is not None:
            self._update_metric(action, reward)
            validated = True
        
        # Propagar feedback
        if reward != 0:
            self._propagate_feedback(
                self._current_state.current_point,
                reward
            )
        
        return validated
    
    # =========================================================================
    # AÇÕES
    # =========================================================================
    
    def _explore_action(self, point: np.ndarray) -> Dict:
        """Ação exploratória aleatória."""
        direction = np.random.randn(len(point))
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        return {
            'type': 'explore',
            'direction': direction,
            'magnitude': 0.2
        }
    
    def _hypothesis_action(self, hypothesis: Dict) -> Dict:
        """Ação para testar hipótese."""
        return {
            'type': 'test_hypothesis',
            'direction': hypothesis.get('direction', np.zeros(512)),
            'magnitude': hypothesis.get('magnitude', 0.1),
            'hypothesis_id': hypothesis.get('id')
        }
    
    def _goal_action(
        self,
        point: np.ndarray,
        goal: np.ndarray
    ) -> Dict:
        """Ação em direção ao goal."""
        # Projetar goal se necessário
        if hasattr(self.bridge, '_project_to_latent') and len(goal) != len(point):
            goal = self.bridge._project_to_latent(goal)
        
        direction = goal - point
        dist = np.linalg.norm(direction)
        
        if dist > 1e-8:
            direction = direction / dist
        
        return {
            'type': 'move_to_goal',
            'direction': direction,
            'magnitude': min(0.3, dist)
        }
    
    def _minimize_f_action(self, point: np.ndarray) -> Dict:
        """Ação para minimizar free energy."""
        # Gradiente numérico de F
        gradient = self._estimate_f_gradient(point)
        
        # Descer gradiente
        direction = -gradient
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        
        return {
            'type': 'minimize_f',
            'direction': direction,
            'magnitude': 0.1
        }
    
    def _execute_action(
        self,
        point: np.ndarray,
        action: Dict
    ) -> np.ndarray:
        """Executa ação e retorna novo ponto."""
        direction = action['direction']
        magnitude = action['magnitude']
        
        # Garantir dimensão correta
        if len(direction) != len(point):
            direction = np.zeros(len(point))
            direction[:len(action['direction'])] = action['direction'][:len(point)]
        
        new_point = point + magnitude * direction
        
        return new_point
    
    # =========================================================================
    # RACIOCÍNIO COMPOSICIONAL
    # =========================================================================
    
    def _detect_local_gaps(self, point: np.ndarray) -> List[Dict]:
        """Detecta gaps no entorno do ponto atual."""
        gaps = []
        
        if self.bridge is None:
            return gaps
        
        # Verificar se há atratores próximos
        try:
            nearest = self.bridge.get_nearest_anchors(point, k=5)
            
            if not nearest:
                # Nenhum atrator = gap
                gaps.append({
                    'type': 'no_attractor',
                    'location': point,
                    'priority': 1.0
                })
            elif nearest[0][1] > 0.5:
                # Atrator mais próximo está longe = gap
                gaps.append({
                    'type': 'distant_attractor',
                    'location': point,
                    'nearest': nearest[0][0].coordinates,
                    'distance': nearest[0][1],
                    'priority': nearest[0][1]
                })
            
            # Verificar conectividade entre atratores
            if len(nearest) >= 2:
                for i in range(len(nearest)-1):
                    a1, d1 = nearest[i]
                    a2, d2 = nearest[i+1]
                    
                    # Ponto médio
                    midpoint = (a1.coordinates + a2.coordinates) / 2
                    mid_energy = self._compute_free_energy(midpoint)
                    
                    if mid_energy > 0.5:
                        gaps.append({
                            'type': 'barrier',
                            'location': midpoint,
                            'source': a1.coordinates,
                            'target': a2.coordinates,
                            'energy': mid_energy,
                            'priority': mid_energy
                        })
        except Exception as e:
            logger.debug(f"Gap detection error: {e}")
        
        # Ordenar por prioridade
        gaps.sort(key=lambda g: g.get('priority', 0), reverse=True)
        
        return gaps
    
    def _generate_geodesic_hypothesis(self, gap: Dict) -> Optional[Dict]:
        """Gera hipótese como caminho geodésico."""
        gap_type = gap.get('type', '')
        
        if gap_type == 'no_attractor':
            # Hipótese: mover para região mais estruturada
            direction = self._find_structure_direction(gap['location'])
            return {
                'id': f"hyp_{gap_type}_{np.random.randint(1000)}",
                'type': 'seek_structure',
                'direction': direction,
                'magnitude': 0.2,
                'confidence': 0.5
            }
        
        elif gap_type == 'distant_attractor':
            # Hipótese: mover em direção ao atrator
            direction = gap['nearest'] - gap['location']
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return {
                'id': f"hyp_{gap_type}_{np.random.randint(1000)}",
                'type': 'approach_attractor',
                'direction': direction,
                'magnitude': min(0.3, gap['distance']),
                'confidence': 1.0 / (1.0 + gap['distance'])
            }
        
        elif gap_type == 'barrier':
            # Hipótese: encontrar caminho ao redor da barreira
            if self.compositional is not None:
                try:
                    result = self.compositional.reason(
                        gap['source'], gap['target']
                    )
                    # Usar direção do caminho
                    if len(result.points) > 1:
                        direction = result.points[1] - result.points[0]
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        return {
                            'id': f"hyp_{gap_type}_{np.random.randint(1000)}",
                            'type': 'bypass_barrier',
                            'direction': direction,
                            'magnitude': 0.15,
                            'confidence': 1.0 / (1.0 + gap['energy']),
                            'path': result.points
                        }
                except:
                    pass
            
            # Fallback: direção perpendicular
            direct = gap['target'] - gap['source']
            perp = np.roll(direct, 1)  # Aproximação perpendicular
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            return {
                'id': f"hyp_{gap_type}_{np.random.randint(1000)}",
                'type': 'bypass_barrier',
                'direction': perp,
                'magnitude': 0.1,
                'confidence': 0.3
            }
        
        return None
    
    def _find_structure_direction(self, point: np.ndarray) -> np.ndarray:
        """Encontra direção de maior estrutura."""
        if self.bridge is None:
            return np.random.randn(len(point))
        
        # Amostrar direções
        best_dir = np.random.randn(len(point))
        best_struct = 0
        
        for _ in range(10):
            direction = np.random.randn(len(point))
            direction = direction / np.linalg.norm(direction)
            
            probe = point + 0.2 * direction
            
            # Medir estrutura = proximidade a atrator
            try:
                nearest = self.bridge.get_nearest_anchors(probe, k=1)
                if nearest:
                    struct = 1.0 / (1.0 + nearest[0][1])
                    if struct > best_struct:
                        best_struct = struct
                        best_dir = direction
            except:
                pass
        
        return best_dir
    
    # =========================================================================
    # APRENDIZADO
    # =========================================================================
    
    def _update_metric(self, action: Dict, reward: float):
        """Atualiza métrica baseado em ação bem-sucedida."""
        point = self._current_state.current_point
        
        update = {
            'point': point.copy(),
            'action': action,
            'reward': reward,
            'iteration': self._current_state.iteration
        }
        
        self._metric_updates.append(update)
        
        # Se bridge suporta deformação
        if hasattr(self.bridge, 'add_deformation'):
            try:
                intensity = self.config.metric_learning_rate * reward
                self.bridge.add_deformation(point, intensity=intensity)
            except:
                pass
    
    def _propagate_feedback(
        self,
        source: np.ndarray,
        reward: float
    ) -> GeodesicFeedback:
        """Propaga feedback ao longo de conexões geodésicas."""
        affected = []
        values = []
        
        # Propagar para pontos recentes na trajetória
        decay = self.config.feedback_decay
        current_value = reward
        
        for i, point in enumerate(reversed(self._trajectory[-self.config.feedback_propagation_depth:])):
            if i > 0:  # Pular o próprio ponto
                affected.append(point)
                values.append(current_value)
                current_value *= decay
        
        return GeodesicFeedback(
            source_point=source,
            affected_points=affected,
            feedback_values=values,
            propagation_path=np.array(affected) if affected else np.array([source]),
            total_impact=sum(values)
        )
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def _compute_free_energy(self, point: np.ndarray) -> float:
        """Calcula free energy no ponto."""
        
        # 1. Tentar usar CompositionalReasoner (já unificado)
        if self.compositional is not None:
            # Se o reasoner tem acesso direto ao field
            if hasattr(self.compositional, 'field') and self.compositional.field is not None:
                if hasattr(self.compositional.field, 'get_free_energy_at'):
                    try:
                        return self.compositional.field.get_free_energy_at(point)
                    except:
                        pass
            
            # Se não, usar a aproximação dele (que pode ter sido melhorada)
            if hasattr(self.compositional, '_approximate_free_energy'):
                try:
                    return self.compositional._approximate_free_energy(point)
                except:
                    pass

        # 2. Fallback: lógica local (duplicada para robustez)
        if self.bridge is None:
            return 0.0
        
        try:
            nearest = self.bridge.get_nearest_anchors(point, k=4)
            if nearest:
                return float(np.mean([d for _, d in nearest]))
        except:
            pass
        
        return 0.0
    
    def _estimate_f_gradient(self, point: np.ndarray) -> np.ndarray:
        """Estima gradiente de F numericamente."""
        eps = 0.01
        grad = np.zeros_like(point)
        
        F0 = self._compute_free_energy(point)
        
        # Amostrar algumas dimensões (para eficiência)
        dims = np.random.choice(len(point), min(50, len(point)), replace=False)
        
        for d in dims:
            point_plus = point.copy()
            point_plus[d] += eps
            F_plus = self._compute_free_energy(point_plus)
            grad[d] = (F_plus - F0) / eps
        
        return grad
    
    def _checkpoint(self, iteration: int):
        """Salva checkpoint do estado."""
        logger.debug(f"Checkpoint at iteration {iteration}: F={self._current_state.free_energy:.4f}")
    
    # =========================================================================
    # API ADICIONAL
    # =========================================================================
    
    def step(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executa um único passo do ciclo.
        
        Útil para integração externa ou debugging.
        """
        if self._current_state is None:
            self._init_cycle(observation, goal)
        else:
            # Atualizar observação
            if hasattr(self.bridge, '_project_to_latent'):
                self._current_state.current_point = self.bridge._project_to_latent(observation)
            else:
                self._current_state.current_point = observation
        
        # Executar fases
        self._perceive_phase()
        hypothesis = self._reason_phase()
        action, reward = self._act_phase(hypothesis, goal)
        validated = self._learn_phase(action, reward)
        
        return {
            'iteration': self._current_state.iteration,
            'free_energy': self._current_state.free_energy,
            'prediction_error': self._current_state.prediction_error,
            'action': action,
            'reward': reward,
            'hypothesis': hypothesis,
            'validated': validated
        }
    
    def reset(self):
        """Reseta estado do loop."""
        self._current_state = None
        self._trajectory = []
        self._metric_updates = []
        self._cumulative_reward = 0.0
    
    def get_state(self) -> Optional[LoopState]:
        """Retorna estado atual."""
        return self._current_state
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Retorna trajetória completa."""
        return self._trajectory.copy()
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "has_compositional": self.compositional is not None,
            "has_self_feeding": self.self_feeding is not None,
            "has_nemesis": self.nemesis is not None,
            "trajectory_length": len(self._trajectory),
            "metric_updates": len(self._metric_updates),
            "cumulative_reward": self._cumulative_reward,
            "current_iteration": self._current_state.iteration if self._current_state else 0,
            "current_free_energy": self._current_state.free_energy if self._current_state else None,
            "config": {
                "max_iterations": self.config.max_iterations,
                "energy_target": self.config.energy_target,
                "exploration_rate": self.config.exploration_rate
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_loop_compositional(
    bridge,
    compositional=None,
    self_feeding_loop=None,
    nemesis=None,
    **config_kwargs
) -> LoopCompositionalIntegration:
    """Factory function."""
    config = LoopCompositionalConfig(**config_kwargs)
    return LoopCompositionalIntegration(
        bridge, compositional, self_feeding_loop, nemesis, config=config
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loop ↔ Compositional Integration - Teste")
    print("=" * 60)
    
    # Importar dependências
    try:
        from vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig
    except ImportError:
        print("ERRO: vqvae_manifold_bridge.py não encontrado")
        exit(1)
    
    # Mock VQ-VAE
    class MockVQVAE:
        def get_codebook(self):
            np.random.seed(42)
            return np.random.randn(4, 256, 128).astype(np.float32)
    
    # Criar bridge
    print("\n1. Criando bridge...")
    bridge = VQVAEManifoldBridge(BridgeConfig(pull_strength=0.5))
    bridge.connect_vqvae(MockVQVAE())
    print(f"   Anchors: {len(bridge.anchor_points)}")
    
    # Criar integração
    print("\n2. Criando integração Loop ↔ Compositional...")
    lci = LoopCompositionalIntegration(bridge, config=LoopCompositionalConfig(
        max_iterations=30,
        hypothesis_generation_interval=5
    ))
    
    # Teste: step individual
    print("\n3. Testando step individual...")
    observation = np.random.randn(384).astype(np.float32)
    goal = np.random.randn(384).astype(np.float32)
    
    step_result = lci.step(observation, goal)
    print(f"   Iteration: {step_result['iteration']}")
    print(f"   Free energy: {step_result['free_energy']:.4f}")
    print(f"   Action type: {step_result['action']['type']}")
    print(f"   Reward: {step_result['reward']:.4f}")
    
    # Teste: ciclo completo
    print("\n4. Testando ciclo autônomo completo...")
    lci.reset()
    
    cycle_result = lci.autonomous_cycle(observation, goal, max_iterations=30)
    
    print(f"   Total iterations: {cycle_result.total_iterations}")
    print(f"   Final F: {cycle_result.final_free_energy:.4f}")
    print(f"   Energy reduction: {cycle_result.energy_reduction:.4f}")
    print(f"   Hypotheses generated: {cycle_result.hypotheses_generated}")
    print(f"   Hypotheses validated: {cycle_result.hypotheses_validated}")
    print(f"   Metric deformations: {cycle_result.metric_deformations}")
    print(f"   Cumulative reward: {cycle_result.cumulative_reward:.4f}")
    print(f"   Duration: {cycle_result.duration_seconds:.2f}s")
    
    # Stats
    print("\n5. Estatísticas:")
    stats = lci.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
