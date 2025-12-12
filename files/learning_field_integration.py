"""
Learning Stack ↔ Field Integration
===================================

Conecta os módulos de Learning (Predictive Coding, Active Inference, Meta-Hebbian)
ao Campo Pré-Estrutural (Manifold, Metric, Geodesic).

Isso permite:
- Predictive Coding operar no espaço curvo
- Active Inference usar distância geodésica
- Meta-Hebbian evoluir regras considerando geometria
- Free Energy calculado com métrica deformada

Teoria:
    F_geometric = D_KL[Q||P] onde a divergência usa métrica g_ij
    
    prediction_error = d_geodesic(observed, predicted)
    
    Δw_hebbian considera curvatura local

Uso:
    from learning_field_integration import LearningFieldIntegration
    
    lfi = LearningFieldIntegration(bridge, predictive_coding, active_inference)
    
    # Predictive coding no manifold
    code, error = lfi.encode_geometric(observation)
    
    # Active inference com geodésicas  
    action = lfi.plan_geometric(current_state)

Autor: G (Alexandria Project)
Versão: 1.0
Fase: 1.2 - Fundação
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class LearningFieldConfig:
    """Configuração da integração Learning ↔ Field."""
    
    # Predictive Coding
    pc_use_geodesic_error: bool = True      # Erro via geodésica
    pc_iterations: int = 5                   # Iterações de settling
    pc_learning_rate: float = 0.1
    
    # Active Inference
    ai_use_geodesic_planning: bool = True
    ai_planning_horizon: int = 5
    ai_num_samples: int = 20
    
    # Meta-Hebbian
    mh_curvature_modulation: bool = True     # Curvatura modula plasticidade
    mh_evolution_interval: int = 100
    
    # Free Energy
    fe_use_geometric_kl: bool = True         # KL com métrica
    fe_temperature: float = 1.0
    
    # Integração
    sync_beliefs: bool = True                # Sincronizar beliefs entre módulos
    propagate_errors: bool = True            # Propagar erros entre níveis


# =============================================================================
# PREDICTIVE CODING GEOMÉTRICO
# =============================================================================

@dataclass
class GeometricPrediction:
    """Predição com informação geométrica."""
    predicted: np.ndarray
    observed: np.ndarray
    error: np.ndarray
    geodesic_error: float            # Erro medido geodesicamente
    euclidean_error: float           # Erro euclidiano (para comparação)
    curvature_at_prediction: float
    precision: float


class GeometricPredictiveCoding:
    """
    Predictive Coding que opera no manifold curvo.
    
    O erro de predição é calculado via distância geodésica,
    não euclidiana. Isso significa que erros "na direção" de
    atratores são ponderados diferentemente.
    """
    
    def __init__(
        self,
        bridge,
        base_pc=None,                 # PredictiveCodingNetwork base
        config: Optional[LearningFieldConfig] = None
    ):
        self.bridge = bridge
        self.base_pc = base_pc
        self.config = config or LearningFieldConfig()
        
        # Estado
        self._beliefs = {}
        self._precisions = {}
    
    def encode(
        self,
        observation: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, GeometricPrediction]:
        """
        Codifica observação via settling no manifold curvo.
        
        Args:
            observation: Embedding observado
            context: Contexto opcional (top-down)
        
        Returns:
            (código, predição_geométrica)
        """
        # Projetar para manifold
        if hasattr(self.bridge, '_project_to_latent'):
            obs_manifold = self.bridge._project_to_latent(observation)
        else:
            obs_manifold = observation
        
        # Usar PC base se disponível
        if self.base_pc is not None:
            code = self.base_pc.encode(observation)
            predicted = self.base_pc.decode(code) if hasattr(self.base_pc, 'decode') else code
        else:
            # Fallback: quantização via bridge
            point = self.bridge.embed(observation)
            code = point.discrete_codes
            predicted = point.coordinates
        
        # Calcular erros
        error = obs_manifold - predicted
        euclidean_error = float(np.linalg.norm(error))
        
        # Erro geodésico
        if self.config.pc_use_geodesic_error:
            geodesic_error = self._geodesic_error(obs_manifold, predicted)
        else:
            geodesic_error = euclidean_error
        
        # Curvatura no ponto de predição
        curvature = self._local_curvature(predicted)
        
        # Precisão adaptativa (maior em regiões curvas = mais certos)
        precision = 1.0 + 0.5 * curvature
        
        pred = GeometricPrediction(
            predicted=predicted,
            observed=obs_manifold,
            error=error,
            geodesic_error=geodesic_error,
            euclidean_error=euclidean_error,
            curvature_at_prediction=curvature,
            precision=precision
        )
        
        return code, pred
    
    def learn(
        self,
        observation: np.ndarray,
        prediction: GeometricPrediction
    ) -> Dict[str, float]:
        """
        Atualiza modelo baseado no erro geométrico.
        """
        metrics = {}
        
        # Erro ponderado por precisão
        weighted_error = prediction.precision * prediction.error
        
        # Atualizar PC base se disponível
        if self.base_pc is not None and hasattr(self.base_pc, 'learn'):
            self.base_pc.learn(observation)
            metrics['base_pc_updated'] = True
        
        # Atualizar beliefs internos
        self._update_beliefs(prediction)
        
        metrics['geodesic_error'] = prediction.geodesic_error
        metrics['euclidean_error'] = prediction.euclidean_error
        metrics['precision'] = prediction.precision
        
        return metrics
    
    def _geodesic_error(
        self,
        observed: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """
        Calcula erro como distância geodésica.
        """
        # Simplificado: usar comprimento do caminho
        # Em produção, usaria GeodesicFlow
        diff = observed - predicted
        
        # Ponderar pela métrica
        try:
            g = self.bridge.compute_metric_deformation(predicted)
            # Distância = sqrt(diff^T @ g @ diff)
            return float(np.sqrt(diff @ g @ diff))
        except:
            return float(np.linalg.norm(diff))
    
    def _local_curvature(self, point: np.ndarray) -> float:
        """
        Curvatura local (traço da métrica - dim).
        """
        try:
            g = self.bridge.compute_metric_deformation(point)
            return float(np.trace(g) - len(point))
        except:
            return 0.0
    
    def _update_beliefs(self, prediction: GeometricPrediction):
        """Atualiza beliefs internos."""
        # Simplificado: média móvel
        key = 'global'
        if key not in self._beliefs:
            self._beliefs[key] = prediction.predicted.copy()
            self._precisions[key] = prediction.precision
        else:
            lr = self.config.pc_learning_rate
            self._beliefs[key] = (1-lr) * self._beliefs[key] + lr * prediction.predicted
            self._precisions[key] = (1-lr) * self._precisions[key] + lr * prediction.precision


# =============================================================================
# ACTIVE INFERENCE GEOMÉTRICO
# =============================================================================

@dataclass
class GeometricPlan:
    """Plano de ação com informação geométrica."""
    actions: List[Dict]
    expected_path: np.ndarray
    expected_free_energy: float
    geodesic_length: float
    information_gain: float
    curvature_traversed: float


class GeometricActiveInference:
    """
    Active Inference que planeja no manifold curvo.
    
    Ações são avaliadas pelo EFE computado geodesicamente.
    O agente prefere caminhos que minimizam surpresa
    considerando a geometria do espaço.
    """
    
    def __init__(
        self,
        bridge,
        base_ai=None,                 # ActiveInferenceAgent base
        compositional=None,           # CompositionalReasoner
        config: Optional[LearningFieldConfig] = None
    ):
        self.bridge = bridge
        self.base_ai = base_ai
        self.compositional = compositional
        self.config = config or LearningFieldConfig()
        
        # Preferências
        self._preferences = None
    
    def plan(
        self,
        current_state: np.ndarray,
        goal: Optional[np.ndarray] = None
    ) -> GeometricPlan:
        """
        Planeja ação minimizando EFE geodésico.
        
        Args:
            current_state: Estado atual
            goal: Goal opcional (senão usa preferências)
        
        Returns:
            GeometricPlan com ações recomendadas
        """
        # Projetar para manifold
        if hasattr(self.bridge, '_project_to_latent'):
            current = self.bridge._project_to_latent(current_state)
        else:
            current = current_state
        
        # Gerar ações candidatas
        candidates = self._generate_action_candidates(current, goal)
        
        # Avaliar cada uma
        best_plan = None
        best_efe = float('inf')
        
        for action in candidates:
            plan = self._evaluate_action(current, action, goal)
            if plan.expected_free_energy < best_efe:
                best_efe = plan.expected_free_energy
                best_plan = plan
        
        return best_plan or GeometricPlan(
            actions=[],
            expected_path=np.array([current]),
            expected_free_energy=float('inf'),
            geodesic_length=0.0,
            information_gain=0.0,
            curvature_traversed=0.0
        )
    
    def _generate_action_candidates(
        self,
        current: np.ndarray,
        goal: Optional[np.ndarray]
    ) -> List[Dict]:
        """Gera ações candidatas."""
        candidates = []
        
        # Projetar goal para mesma dimensão se necessário
        if goal is not None:
            if hasattr(self.bridge, '_project_to_latent') and len(goal) != len(current):
                goal = self.bridge._project_to_latent(goal)
        
        # Ação em direção ao goal
        if goal is not None and len(goal) == len(current):
            candidates.append({
                'type': 'move_to_goal',
                'direction': goal - current,
                'magnitude': min(1.0, np.linalg.norm(goal - current))
            })
        
        # Ações em direção a atratores próximos
        if self.bridge is not None:
            try:
                nearest = self.bridge.get_nearest_anchors(current, k=5)
                for anchor, dist in nearest:
                    direction = anchor.coordinates - current
                    candidates.append({
                        'type': 'move_to_attractor',
                        'direction': direction,
                        'magnitude': min(0.5, dist),
                        'attractor': anchor
                    })
            except:
                pass
        
        # Ação exploratória
        for _ in range(3):
            direction = np.random.randn(len(current))
            direction = direction / np.linalg.norm(direction)
            candidates.append({
                'type': 'explore',
                'direction': direction,
                'magnitude': 0.3
            })
        
        return candidates
    
    def _evaluate_action(
        self,
        current: np.ndarray,
        action: Dict,
        goal: Optional[np.ndarray]
    ) -> GeometricPlan:
        """
        Avalia uma ação computando EFE geodésico.
        """
        direction = action['direction']
        magnitude = action['magnitude']
        
        # Normalizar direção
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-8:
            direction = direction / dir_norm
        
        # Ponto esperado após ação
        expected_end = current + magnitude * direction
        
        # Computar caminho
        if self.compositional is not None:
            try:
                result = self.compositional.reason(current, expected_end)
                path = result.points
                geodesic_length = result.path_length
            except:
                path = np.array([current, expected_end])
                geodesic_length = magnitude
        else:
            path = np.array([current, expected_end])
            geodesic_length = magnitude
        
        # EFE = Risk + Ambiguity
        risk = self._compute_risk(expected_end, goal)
        ambiguity = self._compute_ambiguity(path)
        info_gain = self._compute_information_gain(current, expected_end)
        curvature = self._compute_path_curvature(path)
        
        efe = risk + ambiguity - 0.5 * info_gain
        
        return GeometricPlan(
            actions=[action],
            expected_path=path,
            expected_free_energy=efe,
            geodesic_length=geodesic_length,
            information_gain=info_gain,
            curvature_traversed=curvature
        )
    
    def _compute_risk(
        self,
        state: np.ndarray,
        goal: Optional[np.ndarray]
    ) -> float:
        """Risk = distância ao goal preferido."""
        if goal is None:
            if self._preferences is not None:
                goal = self._preferences
            else:
                return 0.0
        
        return float(np.linalg.norm(state - goal))
    
    def _compute_ambiguity(self, path: np.ndarray) -> float:
        """Ambiguity = incerteza média ao longo do caminho."""
        if self.bridge is None:
            return 1.0
        
        uncertainties = []
        for point in path:
            try:
                nearest = self.bridge.get_nearest_anchors(point, k=1)
                if nearest:
                    _, dist = nearest[0]
                    uncertainties.append(dist)
            except:
                uncertainties.append(1.0)
        
        return float(np.mean(uncertainties)) if uncertainties else 1.0
    
    def _compute_information_gain(
        self,
        current: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """Information gain = redução de incerteza esperada."""
        # Simplificado: maior ganho para regiões de alta curvatura
        try:
            g_current = self.bridge.compute_metric_deformation(current)
            g_expected = self.bridge.compute_metric_deformation(expected)
            
            curv_current = np.trace(g_current) - len(current)
            curv_expected = np.trace(g_expected) - len(expected)
            
            # Ganho se move para região mais estruturada
            return max(0, curv_expected - curv_current)
        except:
            return 0.0
    
    def _compute_path_curvature(self, path: np.ndarray) -> float:
        """Curvatura total ao longo do caminho."""
        if self.bridge is None or len(path) < 2:
            return 0.0
        
        total = 0.0
        for point in path[::max(1, len(path)//5)]:
            try:
                g = self.bridge.compute_metric_deformation(point)
                total += np.trace(g) - len(point)
            except:
                pass
        
        return total
    
    def set_preferences(self, preferences: np.ndarray):
        """Define preferências (goal states)."""
        self._preferences = preferences


# =============================================================================
# META-HEBBIAN GEOMÉTRICO
# =============================================================================

class GeometricMetaHebbian:
    """
    Meta-Hebbian que considera geometria do manifold.
    
    Regras de plasticidade são moduladas pela curvatura local.
    Regiões de alta curvatura (atratores) têm plasticidade diferente.
    """
    
    def __init__(
        self,
        bridge,
        base_mh=None,                 # MetaHebbianPlasticity base
        config: Optional[LearningFieldConfig] = None
    ):
        self.bridge = bridge
        self.base_mh = base_mh
        self.config = config or LearningFieldConfig()
        
        # Regras por região
        self._regional_rules = {}
    
    def compute_update(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        location: np.ndarray
    ) -> np.ndarray:
        """
        Computa update Hebbiano modulado por curvatura.
        
        Args:
            pre: Ativação pré-sináptica
            post: Ativação pós-sináptica
            location: Localização no manifold
        
        Returns:
            Delta de peso
        """
        # Curvatura local
        curvature = self._get_curvature(location)
        
        # Modular learning rate por curvatura
        if self.config.mh_curvature_modulation:
            # Alta curvatura = mais plasticidade (região importante)
            lr_modulation = 1.0 + 0.5 * curvature
        else:
            lr_modulation = 1.0
        
        # Update base
        if self.base_mh is not None:
            delta = self.base_mh.compute_weight_update(
                np.outer(post, pre),  # Matriz de pesos dummy
                pre, post
            )
        else:
            # Regra Hebbiana simples
            delta = np.outer(post, pre) * 0.01
        
        return delta * lr_modulation
    
    def evolve_rules(
        self,
        fitness_scores: List[float],
        locations: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evolui regras considerando localização geométrica.
        """
        metrics = {}
        
        # Agrupar por região de curvatura
        low_curv_fitness = []
        high_curv_fitness = []
        
        for fitness, loc in zip(fitness_scores, locations):
            curv = self._get_curvature(loc)
            if curv > 0.5:
                high_curv_fitness.append(fitness)
            else:
                low_curv_fitness.append(fitness)
        
        metrics['high_curvature_mean_fitness'] = np.mean(high_curv_fitness) if high_curv_fitness else 0
        metrics['low_curvature_mean_fitness'] = np.mean(low_curv_fitness) if low_curv_fitness else 0
        
        # Evoluir regras base
        if self.base_mh is not None and hasattr(self.base_mh, 'evolve_rules'):
            self.base_mh.evolve_rules(fitness_scores)
            metrics['base_evolved'] = True
        
        return metrics
    
    def _get_curvature(self, location: np.ndarray) -> float:
        """Curvatura no ponto."""
        try:
            if hasattr(self.bridge, '_project_to_latent'):
                location = self.bridge._project_to_latent(location)
            g = self.bridge.compute_metric_deformation(location)
            return float(np.trace(g) - len(location))
        except:
            return 0.0


# =============================================================================
# INTEGRAÇÃO UNIFICADA
# =============================================================================

class LearningFieldIntegration:
    """
    Integração unificada de todos os módulos de Learning com Field.
    
    Coordena Predictive Coding, Active Inference, e Meta-Hebbian
    no contexto do manifold curvo.
    """
    
    def __init__(
        self,
        bridge,
        predictive_coding=None,
        active_inference=None,
        meta_hebbian=None,
        compositional=None,
        config: Optional[LearningFieldConfig] = None
    ):
        self.bridge = bridge
        self.config = config or LearningFieldConfig()
        
        # Wrappers geométricos
        self.geometric_pc = GeometricPredictiveCoding(
            bridge, predictive_coding, self.config
        )
        self.geometric_ai = GeometricActiveInference(
            bridge, active_inference, compositional, self.config
        )
        self.geometric_mh = GeometricMetaHebbian(
            bridge, meta_hebbian, self.config
        )
        
        # Estado compartilhado
        self._shared_beliefs = {}
        self._free_energy_history = []
    
    def process_observation(
        self,
        observation: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processa observação através de todo o stack.
        
        1. Predictive Coding: encode e compute error
        2. Update beliefs
        3. Compute free energy
        """
        results = {}
        
        # Predictive Coding
        code, prediction = self.geometric_pc.encode(observation, context)
        results['code'] = code
        results['prediction'] = prediction
        
        # Learn
        learn_metrics = self.geometric_pc.learn(observation, prediction)
        results['learning'] = learn_metrics
        
        # Free energy
        F = prediction.geodesic_error + 0.1 * prediction.curvature_at_prediction
        self._free_energy_history.append(F)
        results['free_energy'] = F
        
        # Sync beliefs se configurado
        if self.config.sync_beliefs:
            self._shared_beliefs['current_state'] = observation
            self._shared_beliefs['current_code'] = code
            self._shared_beliefs['current_prediction'] = prediction.predicted
        
        return results
    
    def plan_action(
        self,
        current_state: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None
    ) -> GeometricPlan:
        """
        Planeja próxima ação via Active Inference geométrico.
        """
        if current_state is None:
            current_state = self._shared_beliefs.get('current_state')
        
        if current_state is None:
            raise ValueError("No current state available")
        
        return self.geometric_ai.plan(current_state, goal)
    
    def update_plasticity(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        location: np.ndarray
    ) -> np.ndarray:
        """
        Computa update de plasticidade Meta-Hebbiano.
        """
        return self.geometric_mh.compute_update(pre, post, location)
    
    def cognitive_cycle(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Ciclo cognitivo completo:
        1. Perceber (PC)
        2. Planejar (AI)
        3. Aprender (MH)
        """
        results = {}
        
        # 1. Perceber
        perception = self.process_observation(observation)
        results['perception'] = perception
        
        # 2. Planejar
        plan = self.plan_action(observation, goal)
        results['plan'] = plan
        
        # 3. Aprender (se erro alto)
        if perception['prediction'].geodesic_error > 0.1:
            # Trigger meta-learning
            fitness = 1.0 / (perception['prediction'].geodesic_error + 0.1)
            self.geometric_mh.evolve_rules(
                [fitness],
                [perception['prediction'].predicted]
            )
            results['meta_learning_triggered'] = True
        
        return results
    
    def get_free_energy_trend(self, window: int = 10) -> float:
        """Retorna tendência do free energy."""
        if len(self._free_energy_history) < window:
            return 0.0
        
        recent = self._free_energy_history[-window:]
        older = self._free_energy_history[-2*window:-window] if len(self._free_energy_history) >= 2*window else recent
        
        return np.mean(recent) - np.mean(older)
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "free_energy_history_size": len(self._free_energy_history),
            "current_free_energy": self._free_energy_history[-1] if self._free_energy_history else None,
            "free_energy_trend": self.get_free_energy_trend(),
            "shared_beliefs_keys": list(self._shared_beliefs.keys()),
            "config": {
                "pc_geodesic_error": self.config.pc_use_geodesic_error,
                "ai_geodesic_planning": self.config.ai_use_geodesic_planning,
                "mh_curvature_modulation": self.config.mh_curvature_modulation
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_learning_field_integration(
    bridge,
    predictive_coding=None,
    active_inference=None,
    meta_hebbian=None,
    compositional=None,
    **config_kwargs
) -> LearningFieldIntegration:
    """Factory function."""
    config = LearningFieldConfig(**config_kwargs)
    return LearningFieldIntegration(
        bridge,
        predictive_coding,
        active_inference,
        meta_hebbian,
        compositional,
        config
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Learning Stack ↔ Field Integration - Teste")
    print("=" * 60)
    
    # Importar dependências
    try:
        from vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig
    except ImportError:
        print("ERRO: vqvae_manifold_bridge.py não encontrado")
        exit(1)
    
    # Mock VQ-VAE
    class MockVQVAE:
        def __init__(self):
            np.random.seed(42)
        def get_codebook(self):
            return np.random.randn(4, 256, 128).astype(np.float32)
    
    # Criar bridge
    print("\n1. Criando bridge...")
    bridge = VQVAEManifoldBridge(BridgeConfig(pull_strength=0.5))
    bridge.connect_vqvae(MockVQVAE())
    print(f"   Anchors: {len(bridge.anchor_points)}")
    
    # Criar integração
    print("\n2. Criando integração Learning ↔ Field...")
    lfi = LearningFieldIntegration(bridge)
    
    # Teste: processar observação
    print("\n3. Processando observação...")
    observation = np.random.randn(384).astype(np.float32)
    
    result = lfi.process_observation(observation)
    
    print(f"   Code shape: {np.array(result['code']).shape}")
    print(f"   Geodesic error: {result['prediction'].geodesic_error:.4f}")
    print(f"   Euclidean error: {result['prediction'].euclidean_error:.4f}")
    print(f"   Curvature: {result['prediction'].curvature_at_prediction:.4f}")
    print(f"   Free energy: {result['free_energy']:.4f}")
    
    # Teste: planejar ação
    print("\n4. Planejando ação...")
    goal = np.random.randn(384).astype(np.float32)
    
    plan = lfi.plan_action(observation, goal)
    
    print(f"   Expected FE: {plan.expected_free_energy:.4f}")
    print(f"   Geodesic length: {plan.geodesic_length:.4f}")
    print(f"   Information gain: {plan.information_gain:.4f}")
    print(f"   Actions: {len(plan.actions)}")
    
    # Teste: ciclo cognitivo completo
    print("\n5. Ciclo cognitivo completo...")
    cycle_result = lfi.cognitive_cycle(observation, goal)
    
    print(f"   Perception FE: {cycle_result['perception']['free_energy']:.4f}")
    print(f"   Plan FE: {cycle_result['plan'].expected_free_energy:.4f}")
    print(f"   Meta-learning: {cycle_result.get('meta_learning_triggered', False)}")
    
    # Stats
    print("\n6. Estatísticas:")
    stats = lfi.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
