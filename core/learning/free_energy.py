"""
Free Energy Principle - Alexandria
===================================

Implementação completa do Princípio de Energia Livre para o sistema Alexandria.
Este é o topo da hierarquia conceitual que unifica todos os módulos anteriores.

O Princípio de Energia Livre (Friston, 2010) afirma que sistemas auto-organizados
resistem à entropia minimizando uma quantidade chamada "energia livre variacional".

                            FREE ENERGY PRINCIPLE
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              PERCEPÇÃO          AÇÃO          APRENDIZADO
             (Predictive     (Active        (Meta-Hebbian)
              Coding)        Inference)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                         MINIMIZAÇÃO DE F
                    F = E_Q[log Q(s) - log P(o,s)]
                    F = Complexity - Accuracy
                    F = Energy - Entropy


Hierarquia completa:
    ✅ Hebbian (base) - minimiza energia local
    ✅ Meta-Hebbian - aprende como minimizar
    ✅ Predictive Coding - minimiza erro de predição
    ✅ Active Inference - minimiza F esperado via ação
    ✅ Free Energy - princípio unificador (ESTE ARQUIVO)


Este módulo implementa:
1. Variational Free Energy (VFE) - métrica central
2. Expected Free Energy (EFE) - para seleção de ação
3. Orquestrador que coordena todos os módulos
4. Self-tuning baseado em gradientes de F
5. Monitoramento de "saúde" do sistema

Referências:
- Friston (2010) - The free-energy principle: a unified brain theory?
- Friston (2019) - A free energy principle for a particular physics
- Parr, Pezzulo & Friston (2022) - Active Inference: The Free Energy Principle in Mind, Brain, and Behavior

Autor: G (Alexandria Project)
Versão: 1.0
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import pickle
import time
from collections import deque

# =============================================================================
# IMPORTS DOS MÓDULOS
# =============================================================================

try:
    from meta_hebbian import MetaHebbianPlasticity, create_meta_hebbian_system
    HAS_META_HEBBIAN = True
except ImportError:
    HAS_META_HEBBIAN = False

try:
    from predictive_coding import PredictiveCodingNetwork, create_predictive_coding_system
    HAS_PREDICTIVE_CODING = True
except ImportError:
    HAS_PREDICTIVE_CODING = False

try:
    from active_inference import (
        ActiveInferenceAgent, 
        ActiveInferenceAlexandria,
        create_active_inference_system,
        Action,
        ActionType
    )
    HAS_ACTIVE_INFERENCE = True
except ImportError:
    HAS_ACTIVE_INFERENCE = False

try:
    from integration_layer import (
        AlexandriaIntegratedPipeline,
        create_integrated_pipeline,
        SparseGraphAdapter
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False


# =============================================================================
# TEORIA: DECOMPOSIÇÃO DA FREE ENERGY
# =============================================================================

"""
VARIATIONAL FREE ENERGY (VFE):
==============================

    F = E_Q[log Q(s) - log P(o,s)]

Onde:
- Q(s) = distribuição aproximada sobre estados (beliefs)
- P(o,s) = modelo generativo (likelihood × prior)
- o = observações
- s = estados latentes

Decomposição 1 (Complexity - Accuracy):
    F = D_KL[Q(s) || P(s)] - E_Q[log P(o|s)]
        \_____________/     \______________/
          Complexity           Accuracy
    
    Complexity: quão distante Q está do prior P(s)
    Accuracy: quão bem o modelo explica as observações

Decomposição 2 (Energy - Entropy):
    F = E_Q[-log P(o,s)] - H[Q(s)]
        \_____________/   \_____/
           Energy         Entropy
    
    Energy: "custo" das observações sob o modelo
    Entropy: incerteza dos beliefs

Para MINIMIZAR F, o sistema pode:
1. PERCEPÇÃO: Atualizar Q(s) para explicar melhor P(o|s)
2. AÇÃO: Mudar o mundo para que o seja mais provável sob P
3. APRENDIZADO: Modificar P para que se ajuste melhor a o

Isso unifica:
- Predictive Coding → minimiza F via percepção
- Active Inference → minimiza E[F] via ação
- Meta-Hebbian → otimiza parâmetros de P


EXPECTED FREE ENERGY (EFE):
===========================

    G(π) = E_Q[F] sob policy π

Decompõe em:
    G(π) = Risk + Ambiguity
    
    Risk = D_KL[Q(o|π) || P(o)]  (divergência de preferências)
    Ambiguity = E_Q[H(o|s,π)]    (incerteza epistêmica)

Active Inference seleciona ações que minimizam G.
"""


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

class FreeEnergyMode(Enum):
    """Modos de operação do sistema"""
    PERCEPTION = auto()      # Só atualiza beliefs
    ACTION = auto()          # Atua no mundo
    LEARNING = auto()        # Atualiza modelo
    FULL = auto()            # Todos simultaneamente


@dataclass
class FreeEnergyConfig:
    """Configuração do sistema de Free Energy"""
    
    # Dimensões
    state_dim: int = 64              # Dimensão do espaço de estados
    observation_dim: int = 384       # Dimensão das observações (embeddings)
    
    # Pesos dos componentes de F
    complexity_weight: float = 1.0   # Peso do termo de complexidade
    accuracy_weight: float = 1.0     # Peso do termo de acurácia
    
    # Pesos para EFE
    risk_weight: float = 1.0
    ambiguity_weight: float = 1.0
    novelty_weight: float = 0.3      # Bonus para exploração
    
    # Dinâmica
    belief_learning_rate: float = 0.1
    model_learning_rate: float = 0.01
    precision_learning_rate: float = 0.001
    
    # Prior preferences (estados desejados)
    preferred_states: Optional[np.ndarray] = None
    
    # Histórico
    history_length: int = 1000
    
    # Persistência
    save_path: str = "data/free_energy_state.pkl"


# =============================================================================
# VARIATIONAL FREE ENERGY
# =============================================================================

class VariationalFreeEnergy:
    """
    Implementação da Energia Livre Variacional.
    
    Esta é a métrica central que todos os módulos trabalham para minimizar.
    
    F = Complexity - Accuracy
    F = D_KL[Q(s) || P(s)] - E_Q[log P(o|s)]
    """
    
    def __init__(self, config: FreeEnergyConfig):
        self.config = config
        
        # Beliefs Q(s): parametrizado como Gaussiana
        self.belief_mean = np.zeros(config.state_dim)
        self.belief_precision = np.ones(config.state_dim)  # 1/variance
        
        # Prior P(s): também Gaussiana
        self.prior_mean = np.zeros(config.state_dim)
        self.prior_precision = np.ones(config.state_dim) * 0.1  # Prior vago
        
        # Likelihood P(o|s): modelo linear + ruído
        # Mapeia state_dim → observation_dim
        self.likelihood_matrix = np.random.randn(
            config.observation_dim, config.state_dim
        ) * 0.1
        self.likelihood_precision = np.ones(config.observation_dim)
        
        # Matriz de projeção inversa (observation → state) para inferência
        self.recognition_matrix = np.random.randn(
            config.state_dim, config.observation_dim
        ) * 0.1
        
        # Preferências P(o): estados de observação preferidos
        if config.preferred_states is not None:
            self.preferred_observations = config.preferred_states
        else:
            self.preferred_observations = np.zeros(config.observation_dim)
        
        # Histórico
        self.F_history: deque = deque(maxlen=config.history_length)
        self.complexity_history: deque = deque(maxlen=config.history_length)
        self.accuracy_history: deque = deque(maxlen=config.history_length)
        
    def compute(
        self,
        observation: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computa Variational Free Energy.
        
        F = Complexity - Accuracy
        
        Returns:
            F: Energia livre total
            components: Breakdown dos termos
        """
        # === COMPLEXITY ===
        # D_KL[Q(s) || P(s)] para Gaussianas
        complexity = self._kl_divergence_gaussian(
            self.belief_mean, 1.0 / self.belief_precision,
            self.prior_mean, 1.0 / self.prior_precision
        )
        
        # === ACCURACY ===
        # E_Q[log P(o|s)]
        obs_error = np.zeros(self.config.state_dim)  # Default
        if observation is not None:
            # Projeta observação para espaço de estados para comparação
            if len(observation) != self.config.state_dim:
                # Usa recognition model para projetar
                projected_obs = self.recognition_matrix @ observation
            else:
                projected_obs = observation
            
            # Erro no espaço de estados
            obs_error = projected_obs - self.belief_mean
            
            # Log-likelihood (Gaussiana)
            accuracy = -0.5 * np.sum(
                self.belief_precision * obs_error**2
            )
            accuracy += 0.5 * np.sum(np.log(self.belief_precision + 1e-10))
        else:
            accuracy = 0.0
        
        # === FREE ENERGY ===
        F = (
            self.config.complexity_weight * complexity -
            self.config.accuracy_weight * accuracy
        )
        
        # Histórico
        self.F_history.append(F)
        self.complexity_history.append(complexity)
        self.accuracy_history.append(accuracy)
        
        return F, {
            'complexity': complexity,
            'accuracy': accuracy,
            'F': F,
            'belief_entropy': self._entropy_gaussian(1.0 / self.belief_precision),
            'prediction_error': float(np.mean(obs_error**2)) if observation is not None else 0
        }
    
    def update_beliefs(
        self,
        observation: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Atualiza beliefs Q(s) para minimizar F.
        
        Isso é equivalente ao que Predictive Coding faz:
        move os beliefs na direção que reduz erro de predição.
        """
        lr = learning_rate or self.config.belief_learning_rate
        
        # Projeta observação para espaço de estados
        if len(observation) != self.config.state_dim:
            projected_obs = self.recognition_matrix @ observation
        else:
            projected_obs = observation
        
        # Erro de predição no espaço de estados
        prediction_error = projected_obs - self.belief_mean
        
        # Gradiente do termo de complexidade
        grad_complexity = self.belief_precision * (self.belief_mean - self.prior_mean)
        
        # Gradiente do termo de acurácia
        grad_accuracy = self.belief_precision * prediction_error
        
        # Gradiente total de F
        grad_F = grad_complexity - grad_accuracy
        
        # Atualização (gradient descent em F)
        self.belief_mean -= lr * grad_F
        
        # Também atualiza precisão dos beliefs (incerteza)
        error_magnitude = np.mean(prediction_error**2)
        precision_update = self.config.precision_learning_rate * (
            1.0 / (error_magnitude + 0.01) - np.mean(self.belief_precision)
        )
        self.belief_precision += precision_update
        self.belief_precision = np.clip(self.belief_precision, 0.1, 100.0)
        
        # Computa F após atualização
        F_new, components = self.compute(observation)
        
        return {
            'F_before': self.F_history[-2] if len(self.F_history) > 1 else float('inf'),
            'F_after': F_new,
            'F_reduction': (self.F_history[-2] - F_new) if len(self.F_history) > 1 else 0,
            'prediction_error': float(np.mean(prediction_error**2)),
            'gradient_norm': float(np.linalg.norm(grad_F)),
            'components': components
        }
    
    def update_model(
        self,
        observation: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Atualiza modelo generativo (recognition matrix) para minimizar F.
        
        Isso é o APRENDIZADO: melhora a projeção observation → state.
        """
        lr = learning_rate or self.config.model_learning_rate
        
        # Projeta observação
        if len(observation) != self.config.state_dim:
            projected_obs = self.recognition_matrix @ observation
        else:
            projected_obs = observation
        
        # Erro
        prediction_error = projected_obs - self.belief_mean
        
        # Gradiente para recognition matrix
        if len(observation) != self.config.state_dim:
            grad_R = np.outer(prediction_error, observation)
            self.recognition_matrix -= lr * grad_R
        
        return {
            'model_change': float(np.mean(np.abs(prediction_error))),
            'belief_norm': float(np.linalg.norm(self.belief_mean))
        }
    
    def _kl_divergence_gaussian(
        self,
        mu1: np.ndarray, var1: np.ndarray,
        mu2: np.ndarray, var2: np.ndarray
    ) -> float:
        """KL divergence entre duas Gaussianas diagonais"""
        k = len(mu1)
        
        # D_KL = 0.5 * (tr(Σ2^-1 @ Σ1) + (μ2-μ1).T @ Σ2^-1 @ (μ2-μ1) - k + log(det(Σ2)/det(Σ1)))
        var1 = np.maximum(var1, 1e-10)
        var2 = np.maximum(var2, 1e-10)
        
        trace_term = np.sum(var1 / var2)
        mean_term = np.sum((mu2 - mu1)**2 / var2)
        log_det_term = np.sum(np.log(var2)) - np.sum(np.log(var1))
        
        kl = 0.5 * (trace_term + mean_term - k + log_det_term)
        return max(0, kl)  # KL é sempre >= 0
    
    def _entropy_gaussian(self, variance: np.ndarray) -> float:
        """Entropia de Gaussiana diagonal"""
        k = len(variance)
        return 0.5 * k * (1 + np.log(2 * np.pi)) + 0.5 * np.sum(np.log(variance + 1e-10))
    
    def get_belief_state(self) -> Dict[str, np.ndarray]:
        """Retorna estado atual dos beliefs"""
        return {
            'mean': self.belief_mean.copy(),
            'precision': self.belief_precision.copy(),
            'variance': 1.0 / self.belief_precision
        }
    
    def get_surprise(self, observation: np.ndarray) -> float:
        """
        Computa "surpresa" de uma observação.
        
        Surprise = -log P(o) ≈ F (sob certas condições)
        """
        F, _ = self.compute(observation)
        return F


# =============================================================================
# EXPECTED FREE ENERGY (para seleção de ação)
# =============================================================================

class ExpectedFreeEnergy:
    """
    Expected Free Energy para seleção de ações.
    
    G(π) = Risk + Ambiguity
         = D_KL[Q(o|π) || P(o)] + E_Q[H(o|s,π)]
    
    Risk: quão longe das preferências a ação nos leva
    Ambiguity: quanta incerteza sobre outcomes
    """
    
    def __init__(
        self,
        vfe: VariationalFreeEnergy,
        config: FreeEnergyConfig
    ):
        self.vfe = vfe
        self.config = config
        
        # Modelos de transição por tipo de ação
        self.transition_models: Dict[str, np.ndarray] = {}
        self._init_transition_models()
        
    def _init_transition_models(self):
        """Inicializa modelos de transição simplificados"""
        dim = self.config.state_dim
        
        # Ações básicas (pode ser expandido)
        actions = ['explore', 'exploit', 'query', 'consolidate', 'rest']
        
        for action in actions:
            # Matriz de transição inicial
            if action == 'explore':
                # Exploração aumenta variância
                T = np.eye(dim) * 0.8 + np.random.randn(dim, dim) * 0.2
            elif action == 'exploit':
                # Exploitação é mais determinística
                T = np.eye(dim) * 0.95 + np.random.randn(dim, dim) * 0.05
            elif action == 'consolidate':
                # Consolidação move em direção ao prior
                T = np.eye(dim) * 0.9
            else:
                # Default
                T = np.eye(dim) * 0.85 + np.random.randn(dim, dim) * 0.1
            
            self.transition_models[action] = T
    
    def compute(
        self,
        action: str,
        current_state: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computa Expected Free Energy para uma ação.
        
        Returns:
            G: Expected Free Energy (menor é melhor)
            components: Breakdown
        """
        state = current_state if current_state is not None else self.vfe.belief_mean
        
        # Prediz próximo estado
        T = self.transition_models.get(action, np.eye(len(state)))
        predicted_state = T @ state
        
        # Prediz observação
        predicted_obs = self.vfe.likelihood_matrix @ predicted_state
        
        # === RISK ===
        # Divergência das preferências
        risk = np.sum((predicted_obs - self.vfe.preferred_observations)**2)
        risk *= self.config.risk_weight
        
        # === AMBIGUITY ===
        # Incerteza sobre outcomes (simplificado)
        # Em modelo completo, seria H(o|s,π)
        state_uncertainty = 1.0 / np.mean(self.vfe.belief_precision)
        ambiguity = state_uncertainty * self.config.ambiguity_weight
        
        # === NOVELTY BONUS ===
        novelty = 0.0
        if action == 'explore':
            novelty = -self.config.novelty_weight  # Bonus negativo (reduz G)
        
        # === EFE TOTAL ===
        G = risk + ambiguity + novelty
        
        return G, {
            'risk': risk,
            'ambiguity': ambiguity,
            'novelty': novelty,
            'G': G
        }
    
    def select_action(
        self,
        available_actions: Optional[List[str]] = None,
        temperature: float = 1.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Seleciona ação que minimiza G.
        
        Usa softmax para exploração.
        """
        actions = available_actions or list(self.transition_models.keys())
        
        # Computa G para cada ação
        Gs = []
        components_list = []
        for action in actions:
            G, components = self.compute(action)
            Gs.append(G)
            components_list.append(components)
        
        Gs = np.array(Gs)
        
        # Softmax selection (menor G = maior probabilidade)
        Gs_normalized = Gs - np.min(Gs)
        probs = np.exp(-Gs_normalized / temperature)
        probs = probs / np.sum(probs)
        
        # Seleciona
        selected_idx = np.random.choice(len(actions), p=probs)
        selected_action = actions[selected_idx]
        
        return selected_action, {
            'all_actions': list(zip(actions, Gs.tolist())),
            'selected_G': Gs[selected_idx],
            'selection_prob': probs[selected_idx],
            'components': components_list[selected_idx]
        }


# =============================================================================
# ORQUESTRADOR FREE ENERGY
# =============================================================================

class FreeEnergyOrchestrator:
    """
    Orquestrador central que coordena todos os módulos sob o Princípio de Free Energy.
    
    Este é o "cérebro central" que:
    1. Monitora F global do sistema
    2. Decide quando fazer percepção, ação ou aprendizado
    3. Balanceia exploration vs exploitation
    4. Mantém o sistema em regime de baixa energia livre
    """
    
    def __init__(
        self,
        config: Optional[FreeEnergyConfig] = None,
        pipeline: Optional['AlexandriaIntegratedPipeline'] = None
    ):
        self.config = config or FreeEnergyConfig()
        self.pipeline = pipeline
        
        # Componentes core
        self.vfe = VariationalFreeEnergy(self.config)
        self.efe = ExpectedFreeEnergy(self.vfe, self.config)
        
        # Estado
        self.mode = FreeEnergyMode.FULL
        self.timestep = 0
        self.last_F = float('inf')
        
        # Estatísticas
        self.action_counts: Dict[str, int] = {}
        self.mode_history: List[FreeEnergyMode] = []
        
        # Thresholds adaptativos
        self.F_threshold_perception = 10.0  # Quando F > threshold, foca em percepção
        self.F_threshold_action = 5.0       # Quando F moderado, pode agir
        self.F_target = 1.0                 # F alvo (nunca chega a zero)
        
    def step(
        self,
        observation: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa um passo do ciclo de Free Energy.
        
        1. Computa F atual
        2. Decide modo (percepção/ação/aprendizado)
        3. Executa operação
        4. Atualiza estado
        
        Returns:
            result: Resultado do passo
        """
        result = {
            'timestep': self.timestep,
            'mode': None,
            'F_before': None,
            'F_after': None,
            'action_taken': None
        }
        
        # 1. Computa F inicial
        F_before, components = self.vfe.compute(observation)
        result['F_before'] = F_before
        result['components'] = components
        
        # 2. Decide modo baseado em F
        mode = self._select_mode(F_before, observation is not None)
        result['mode'] = mode.name
        self.mode_history.append(mode)
        
        # 3. Executa operação correspondente
        if mode == FreeEnergyMode.PERCEPTION and observation is not None:
            # Atualiza beliefs para explicar observação
            update_result = self.vfe.update_beliefs(observation)
            result['perception'] = update_result
            
        elif mode == FreeEnergyMode.LEARNING and observation is not None:
            # Atualiza modelo generativo
            update_result = self.vfe.update_beliefs(observation)
            model_result = self.vfe.update_model(observation)
            result['learning'] = {**update_result, **model_result}
            
        elif mode == FreeEnergyMode.ACTION:
            # Seleciona e "executa" ação
            action, action_info = self.efe.select_action()
            result['action_taken'] = action
            result['action_info'] = action_info
            
            # Registra
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            
            # Simula efeito da ação no estado
            T = self.efe.transition_models.get(action, np.eye(self.config.state_dim))
            self.vfe.belief_mean = T @ self.vfe.belief_mean
            
        elif mode == FreeEnergyMode.FULL and observation is not None:
            # Faz tudo
            update_result = self.vfe.update_beliefs(observation)
            model_result = self.vfe.update_model(observation)
            action, action_info = self.efe.select_action()
            
            result['perception'] = update_result
            result['learning'] = model_result
            result['action_taken'] = action
            result['action_info'] = action_info
        
        # 4. Computa F final
        F_after, _ = self.vfe.compute(observation)
        result['F_after'] = F_after
        result['F_reduction'] = F_before - F_after
        
        # Atualiza estado
        self.last_F = F_after
        self.timestep += 1
        
        # Adapta thresholds
        self._adapt_thresholds()
        
        return result
    
    def _select_mode(self, F: float, has_observation: bool) -> FreeEnergyMode:
        """
        Seleciona modo de operação baseado em F.
        
        - F muito alto → foca em percepção (entender o que está acontecendo)
        - F moderado → pode agir (mudar o mundo)
        - F baixo → aprendizado (refinar modelo)
        """
        if not has_observation:
            return FreeEnergyMode.ACTION
        
        if F > self.F_threshold_perception:
            return FreeEnergyMode.PERCEPTION
        elif F > self.F_threshold_action:
            # Alterna entre ação e percepção
            if self.timestep % 3 == 0:
                return FreeEnergyMode.ACTION
            else:
                return FreeEnergyMode.PERCEPTION
        else:
            # F baixo: pode fazer tudo ou focar em aprendizado
            if self.timestep % 5 == 0:
                return FreeEnergyMode.LEARNING
            else:
                return FreeEnergyMode.FULL
    
    def _adapt_thresholds(self):
        """Adapta thresholds baseado no histórico de F"""
        if len(self.vfe.F_history) < 10:
            return
        
        recent_F = list(self.vfe.F_history)[-50:]
        mean_F = np.mean(recent_F)
        std_F = np.std(recent_F)
        
        # Ajusta thresholds para serem relativos à distribuição de F
        self.F_threshold_perception = mean_F + std_F
        self.F_threshold_action = mean_F
        self.F_target = max(0.1, mean_F - std_F)
    
    def run(
        self,
        observations: List[np.ndarray],
        callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Roda o orquestrador em uma sequência de observações.
        """
        results = []
        
        for obs in observations:
            result = self.step(observation=obs)
            results.append(result)
            
            if callback:
                callback(result)
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Retorna "saúde" do sistema baseado em métricas de Free Energy.
        """
        if len(self.vfe.F_history) == 0:
            return {'status': 'INITIALIZING', 'F': None}
        
        recent_F = list(self.vfe.F_history)[-100:]
        mean_F = np.mean(recent_F)
        std_F = np.std(recent_F)
        trend = np.polyfit(range(len(recent_F)), recent_F, 1)[0] if len(recent_F) > 1 else 0
        
        # Diagnóstico
        if mean_F < self.F_target:
            status = "OPTIMAL"
            diagnosis = "Sistema em equilíbrio de baixa energia livre"
        elif mean_F < self.F_threshold_action:
            status = "HEALTHY"
            diagnosis = "Sistema funcionando bem, F moderado"
        elif mean_F < self.F_threshold_perception:
            status = "STRESSED"
            diagnosis = "F elevado, sistema precisa de mais percepção"
        else:
            status = "CRITICAL"
            diagnosis = "F muito alto, sistema em dificuldade"
        
        # Trend
        if trend < -0.01:
            trend_status = "IMPROVING"
        elif trend > 0.01:
            trend_status = "DEGRADING"
        else:
            trend_status = "STABLE"
        
        return {
            'status': status,
            'diagnosis': diagnosis,
            'F_current': self.last_F,
            'F_mean': mean_F,
            'F_std': std_F,
            'F_trend': trend_status,
            'timestep': self.timestep,
            'action_distribution': self.action_counts,
            'mode_recent': [m.name for m in self.mode_history[-10:]],
            'thresholds': {
                'perception': self.F_threshold_perception,
                'action': self.F_threshold_action,
                'target': self.F_target
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Retorna estado completo para persistência"""
        return {
            'timestep': self.timestep,
            'last_F': self.last_F,
            'belief_mean': self.vfe.belief_mean,
            'belief_precision': self.vfe.belief_precision,
            'prior_mean': self.vfe.prior_mean,
            'prior_precision': self.vfe.prior_precision,
            'likelihood_matrix': self.vfe.likelihood_matrix,
            'likelihood_precision': self.vfe.likelihood_precision,
            'recognition_matrix': self.vfe.recognition_matrix,
            'transition_models': self.efe.transition_models,
            'action_counts': self.action_counts,
            'F_history': list(self.vfe.F_history),
            'thresholds': {
                'perception': self.F_threshold_perception,
                'action': self.F_threshold_action,
                'target': self.F_target
            }
        }
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Salva estado"""
        path = path or self.config.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.get_state(), f)
        
        return path
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Carrega estado"""
        path = path or self.config.save_path
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.timestep = state['timestep']
            self.last_F = state['last_F']
            self.vfe.belief_mean = state['belief_mean']
            self.vfe.belief_precision = state['belief_precision']
            self.vfe.prior_mean = state['prior_mean']
            self.vfe.prior_precision = state['prior_precision']
            self.vfe.likelihood_matrix = state['likelihood_matrix']
            self.vfe.likelihood_precision = state['likelihood_precision']
            if 'recognition_matrix' in state:
                self.vfe.recognition_matrix = state['recognition_matrix']
            self.efe.transition_models = state['transition_models']
            self.action_counts = state.get('action_counts', {})
            
            thresholds = state.get('thresholds', {})
            self.F_threshold_perception = thresholds.get('perception', 10.0)
            self.F_threshold_action = thresholds.get('action', 5.0)
            self.F_target = thresholds.get('target', 1.0)
            
            return True
        except Exception as e:
            print(f"Erro ao carregar Free Energy: {e}")
            return False


# =============================================================================
# INTEGRAÇÃO COMPLETA COM ALEXANDRIA
# =============================================================================

class AlexandriaFreeEnergySystem:
    """
    Sistema completo de Free Energy para Alexandria.
    
    Integra:
    - FreeEnergyOrchestrator (este arquivo)
    - AlexandriaIntegratedPipeline (integration_layer.py)
    - Todos os módulos anteriores
    
    Este é o ponto de entrada unificado para o sistema cognitivo completo.
    """
    
    def __init__(
        self,
        config: Optional[FreeEnergyConfig] = None,
        embedding_model: Optional[Any] = None,
        vqvae: Optional[Any] = None,
        mycelial: Optional[Any] = None
    ):
        self.config = config or FreeEnergyConfig()
        
        # Cria pipeline integrado
        self.pipeline = None
        if HAS_INTEGRATION:
            self.pipeline = create_integrated_pipeline(
                embedding_model=embedding_model,
                vqvae=vqvae,
                mycelial=mycelial,
                load_existing=True
            )
        
        # Cria orquestrador
        self.orchestrator = FreeEnergyOrchestrator(self.config, self.pipeline)
        
        # Estado
        self.total_observations = 0
        
        print("🧠 Alexandria Free Energy System inicializado")
        print(f"   Dimensões: state={self.config.state_dim}, obs={self.config.observation_dim}")
        
    def process(
        self,
        input_data: Union[str, np.ndarray],
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Processa input através do sistema completo.
        
        Args:
            input_data: Texto ou embedding
            learn: Se True, atualiza todos os módulos
            
        Returns:
            result: Resultado completo do processamento
        """
        result = {
            'timestamp': time.time(),
            'observation_id': self.total_observations
        }
        
        # 1. Obtém embedding
        if isinstance(input_data, str):
            if self.pipeline is not None:
                pipeline_result = self.pipeline.process_text(input_data, learn=learn)
                embedding = self.pipeline._get_embedding(input_data)
                result['pipeline'] = pipeline_result
            else:
                # Fallback
                np.random.seed(hash(input_data) % (2**32))
                embedding = np.random.randn(self.config.observation_dim)
        else:
            embedding = input_data
            if self.pipeline is not None:
                pipeline_result = self.pipeline.process_embedding(embedding, learn=learn)
                result['pipeline'] = pipeline_result
        
        # 2. Projeta para espaço do orquestrador se necessário
        if len(embedding) != self.config.state_dim:
            # Projeção simples via pooling
            projected = self._project_observation(embedding)
        else:
            projected = embedding
        
        # 3. Passa pelo orquestrador de Free Energy
        fe_result = self.orchestrator.step(observation=projected)
        result['free_energy'] = fe_result
        
        # 4. Extrai métricas chave
        result['F'] = fe_result['F_after']
        result['mode'] = fe_result['mode']
        result['action'] = fe_result.get('action_taken')
        
        self.total_observations += 1
        
        return result
    
    def _project_observation(self, obs: np.ndarray) -> np.ndarray:
        """Projeta observação para dimensão do espaço de estados"""
        target_dim = self.config.state_dim
        source_dim = len(obs)
        
        if source_dim == target_dim:
            return obs
        elif source_dim > target_dim:
            # Pooling
            chunk_size = source_dim // target_dim
            reshaped = obs[:chunk_size * target_dim].reshape(target_dim, chunk_size)
            return reshaped.mean(axis=1)
        else:
            # Padding
            padded = np.zeros(target_dim)
            padded[:source_dim] = obs
            return padded
    
    def get_system_status(self) -> Dict[str, Any]:
        """Status completo do sistema"""
        status = {
            'total_observations': self.total_observations,
            'free_energy': self.orchestrator.get_system_health()
        }
        
        if self.pipeline is not None:
            status['pipeline'] = self.pipeline.get_system_status()
        
        return status
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        Obtém recomendação do sistema sobre próxima ação.
        
        Combina Active Inference com análise de Free Energy.
        """
        health = self.orchestrator.get_system_health()
        
        # Seleciona ação via EFE
        action, action_info = self.orchestrator.efe.select_action()
        
        # Gera explicação
        explanation = self._generate_explanation(health, action, action_info)
        
        return {
            'recommended_action': action,
            'action_info': action_info,
            'system_status': health['status'],
            'explanation': explanation,
            'F_current': health['F_current']
        }
    
    def _generate_explanation(
        self,
        health: Dict,
        action: str,
        action_info: Dict
    ) -> str:
        """Gera explicação em linguagem natural"""
        status = health['status']
        
        explanations = {
            'explore': f"Sistema sugere EXPLORAR. Status: {status}. "
                      f"Buscar novos conhecimentos para reduzir incerteza.",
            'exploit': f"Sistema sugere EXPLOITAR. Status: {status}. "
                      f"Aprofundar em áreas já conhecidas.",
            'query': f"Sistema sugere QUERY. Status: {status}. "
                    f"Fazer busca direcionada para preencher gaps.",
            'consolidate': f"Sistema sugere CONSOLIDAR. Status: {status}. "
                          f"Integrar conhecimentos antes de expandir.",
            'rest': f"Sistema sugere PAUSAR. Status: {status}. "
                   f"Permitir settling dos estados internos."
        }
        
        return explanations.get(action, f"Ação: {action}. Status: {status}")
    
    def save_state(self) -> str:
        """Salva estado completo"""
        self.orchestrator.save_state()
        if self.pipeline is not None:
            self.pipeline.save_state()
        return self.config.save_path
    
    def load_state(self) -> bool:
        """Carrega estado"""
        loaded = self.orchestrator.load_state()
        if self.pipeline is not None:
            self.pipeline.load_state()
        return loaded


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_free_energy_system(
    state_dim: int = 64,
    observation_dim: int = 384,
    embedding_model: Optional[Any] = None,
    vqvae: Optional[Any] = None,
    mycelial: Optional[Any] = None,
    load_existing: bool = True
) -> AlexandriaFreeEnergySystem:
    """
    Factory function para criar sistema completo de Free Energy.
    """
    config = FreeEnergyConfig(
        state_dim=state_dim,
        observation_dim=observation_dim
    )
    
    system = AlexandriaFreeEnergySystem(
        config=config,
        embedding_model=embedding_model,
        vqvae=vqvae,
        mycelial=mycelial
    )
    
    if load_existing:
        loaded = system.load_state()
        if loaded:
            print(f"✅ Free Energy System carregado: timestep {system.orchestrator.timestep}")
        else:
            print("🌱 Free Energy System inicializado fresh")
    
    return system


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FREE ENERGY PRINCIPLE - ALEXANDRIA")
    print("=" * 70)
    
    # Criar sistema
    config = FreeEnergyConfig(state_dim=64, observation_dim=64)
    orchestrator = FreeEnergyOrchestrator(config)
    
    # Simular observações
    print("\n🔄 SIMULANDO CICLOS DE FREE ENERGY...")
    
    for i in range(30):
        # Observação simulada
        obs = np.random.randn(64)
        
        # Passo do orquestrador
        result = orchestrator.step(observation=obs)
        
        if i % 5 == 0:
            print(f"\n   Timestep {result['timestep']}:")
            print(f"      Mode: {result['mode']}")
            print(f"      F: {result['F_before']:.3f} → {result['F_after']:.3f}")
            if result.get('action_taken'):
                print(f"      Action: {result['action_taken']}")
    
    # Health check
    print("\n📊 SAÚDE DO SISTEMA:")
    health = orchestrator.get_system_health()
    print(f"   Status: {health['status']}")
    print(f"   Diagnóstico: {health['diagnosis']}")
    print(f"   F médio: {health['F_mean']:.3f} ± {health['F_std']:.3f}")
    print(f"   Tendência: {health['F_trend']}")
    print(f"   Ações: {health['action_distribution']}")
    
    # Teste do sistema completo
    print("\n🧠 TESTANDO SISTEMA COMPLETO...")
    
    # Usa dimensões compatíveis com o pipeline (384D para observações)
    full_config = FreeEnergyConfig(state_dim=64, observation_dim=384)
    system = AlexandriaFreeEnergySystem(full_config)
    
    for i in range(10):
        # Observação com dimensão correta para o pipeline
        obs = np.random.randn(384)
        obs = obs / np.linalg.norm(obs)  # Normaliza
        result = system.process(obs)
        
        if i % 3 == 0:
            print(f"   Obs {i}: F={result['F']:.3f}, mode={result['mode']}")
    
    # Recomendação
    print("\n🎯 RECOMENDAÇÃO:")
    rec = system.get_recommendation()
    print(f"   Ação: {rec['recommended_action']}")
    print(f"   Explicação: {rec['explanation']}")
    
    # Salvar
    save_path = system.save_state()
    print(f"\n💾 Estado salvo em: {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ FREE ENERGY PRINCIPLE IMPLEMENTADO")
    print("=" * 70)
    
    print("""
    
HIERARQUIA COMPLETA:
====================

    ┌────────────────────────────────────────────────────────────────┐
    │                    FREE ENERGY PRINCIPLE                       │
    │                                                                │
    │   F = Complexity - Accuracy                                    │
    │   F = D_KL[Q(s)||P(s)] - E_Q[log P(o|s)]                      │
    │                                                                │
    │   Minimizar F unifica TUDO:                                    │
    │                                                                │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
    │   │ PERCEPÇÃO   │  │   AÇÃO      │  │ APRENDIZADO │           │
    │   │             │  │             │  │             │           │
    │   │ Atualiza    │  │ Muda o      │  │ Melhora     │           │
    │   │ Q(s) para   │  │ mundo para  │  │ P(o,s)      │           │
    │   │ explicar o  │  │ o ser mais  │  │ para se     │           │
    │   │             │  │ provável    │  │ ajustar     │           │
    │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
    │          │                │                │                   │
    │          ▼                ▼                ▼                   │
    │   ┌─────────────────────────────────────────────────┐         │
    │   │           ORQUESTRADOR DE FREE ENERGY           │         │
    │   │                                                 │         │
    │   │  • Monitora F global                           │         │
    │   │  • Decide modo (percepção/ação/aprendizado)    │         │
    │   │  • Adapta thresholds dinamicamente             │         │
    │   │  • Reporta "saúde" do sistema                  │         │
    │   └─────────────────────────────────────────────────┘         │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                   MÓDULOS IMPLEMENTADOS                        │
    │                                                                │
    │   ✅ Hebbian          - Energia local                         │
    │   ✅ Meta-Hebbian     - Aprende regras de aprendizado         │
    │   ✅ Predictive Coding - Minimiza erro de predição            │
    │   ✅ Active Inference  - Age para minimizar E[F]              │
    │   ✅ Free Energy       - Princípio unificador                 │
    │   ✅ Integration Layer - Cola tudo                            │
    └────────────────────────────────────────────────────────────────┘


STATUS DO SISTEMA:
==================

    OPTIMAL    : F < target          (equilíbrio)
    HEALTHY    : F < threshold_action (funcionando bem)
    STRESSED   : F < threshold_perc   (precisa percepção)
    CRITICAL   : F > threshold_perc   (em dificuldade)


USO:
====

    from free_energy import create_free_energy_system
    
    # Criar sistema completo
    system = create_free_energy_system(
        embedding_model=model,
        vqvae=monolith,
        mycelial=mycelial_net
    )
    
    # Processar
    result = system.process("Vector quantization paper...")
    print(f"Free Energy: {result['F']}")
    print(f"Mode: {result['mode']}")
    
    # Status
    status = system.get_system_status()
    print(f"Health: {status['free_energy']['status']}")
    
    # Recomendação
    rec = system.get_recommendation()
    print(f"Sugestão: {rec['recommended_action']}")
    print(f"Explicação: {rec['explanation']}")
    
    """)
