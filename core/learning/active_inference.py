"""
Active Inference Module for Alexandria
=======================================

Implementação completa de Active Inference para o sistema Alexandria.
Baseado em: Karl Friston's Free Energy Principle e Active Inference Framework.

Active Inference = Predictive Coding + AÇÃO

O sistema não só prediz passivamente, mas ATUA no mundo para:
1. Confirmar predições (exploração epistêmica)
2. Reduzir incerteza (information seeking)
3. Alcançar estados desejados (goal-directed behavior)

Para Alexandria, ações incluem:
- Gerar queries de busca
- Navegar conexões myceliais
- Priorizar papers para leitura
- Identificar gaps no conhecimento
- Sugerir papers relacionados

Hierarquia de paradigmas:
    Hebbian → Meta-Hebbian → Predictive Coding → Active Inference → Free Energy
                                                        ↑
                                                   VOCÊ ESTÁ AQUI

Equação fundamental:
    G(π) = E_Q[log Q(s) - log P(s,o|π)]
    
    Onde:
    - G(π) = Expected Free Energy da policy π
    - Q(s) = Beliefs sobre estados
    - P(s,o|π) = Modelo generativo com policy
    
    Decompõe em:
    G(π) = Risk + Ambiguity
         = D_KL[Q(o|π)||P(o)] + E_Q[H(o|s,π)]
         
    Risk: divergência de preferências
    Ambiguity: incerteza sobre outcomes

Referências:
- Friston et al. (2017) - Active Inference: A Process Theory
- Parr & Friston (2019) - Generalised free energy and active inference
- Da Costa et al. (2020) - Active inference on discrete state-spaces

Autor: G (Alexandria Project)
Versão: 1.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import pickle
from pathlib import Path
from collections import defaultdict
import heapq

# Imports dos módulos anteriores
try:
    from predictive_coding import PredictiveCodingNetwork, create_predictive_coding_system
except ImportError:
    PredictiveCodingNetwork = None

try:
    from meta_hebbian import MetaHebbianPlasticity
except ImportError:
    MetaHebbianPlasticity = None


# =============================================================================
# TIPOS E CONFIGURAÇÃO
# =============================================================================

class ActionType(Enum):
    """Tipos de ações que o agente pode tomar"""
    QUERY_SEARCH = auto()       # Buscar papers com query específica
    EXPLORE_CLUSTER = auto()    # Explorar cluster de conceitos
    FOLLOW_CONNECTION = auto()  # Seguir conexão mycelial
    DEEPEN_TOPIC = auto()       # Aprofundar em tópico específico
    BRIDGE_CONCEPTS = auto()    # Buscar conexão entre conceitos
    FILL_GAP = auto()           # Preencher gap identificado
    CONSOLIDATE = auto()        # Consolidar conhecimento existente
    REST = auto()               # Não fazer nada (observar)


@dataclass
class Action:
    """Representa uma ação concreta"""
    action_type: ActionType
    target: str                          # Query, cluster_id, concept, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_information_gain: float = 0.0
    expected_risk: float = 0.0
    priority: float = 0.0
    
    def __lt__(self, other):
        """Para uso em priority queue"""
        return self.priority > other.priority  # Maior prioridade primeiro


@dataclass 
class Belief:
    """Estado de crença sobre um conceito/região"""
    concept_id: str
    mean: np.ndarray                     # μ - estimativa atual
    precision: np.ndarray                # Π - confiança (inverso da variância)
    last_updated: int = 0                # Timestamp da última atualização
    observation_count: int = 0           # Quantas vezes observado
    
    @property
    def uncertainty(self) -> float:
        """Incerteza total (menor precisão = maior incerteza)"""
        return 1.0 / (np.mean(self.precision) + 0.01)


@dataclass
class Preference:
    """Preferências do agente (estados desejados)"""
    topic_weights: Dict[str, float] = field(default_factory=dict)
    novelty_preference: float = 0.5      # 0 = exploitar, 1 = explorar
    depth_preference: float = 0.5        # 0 = breadth, 1 = depth
    connection_preference: float = 0.5   # Preferência por conexões


@dataclass
class ActiveInferenceConfig:
    """Configuração do sistema de Active Inference"""
    
    # Dimensões
    state_dim: int = 64                  # Dimensão do estado latente
    action_embedding_dim: int = 32       # Dimensão do embedding de ações
    
    # Planning
    planning_horizon: int = 5            # Quantos passos à frente planejar
    num_action_samples: int = 20         # Ações amostradas por passo
    temperature: float = 1.0             # Softmax temperature para seleção
    
    # Free Energy weights
    risk_weight: float = 1.0             # Peso do termo de risco
    ambiguity_weight: float = 1.0        # Peso do termo de ambiguidade
    novelty_bonus: float = 0.3           # Bônus para exploração
    
    # Aprendizado
    learning_rate: float = 0.01
    belief_decay: float = 0.99           # Decay da precisão ao longo do tempo
    
    # Persistência
    save_path: str = "data/active_inference_state.pkl"


# =============================================================================
# MODELO GENERATIVO
# =============================================================================

class GenerativeModel:
    """
    Modelo generativo P(o, s, π) para Active Inference.
    
    Componentes:
    - P(s_t+1 | s_t, a_t): Dinâmica de transição
    - P(o_t | s_t): Modelo de observação (likelihood)
    - P(s_0): Prior sobre estados iniciais
    - P(π): Prior sobre policies
    
    Para Alexandria:
    - Estados = representações latentes de conhecimento
    - Observações = embeddings de papers/chunks
    - Ações = queries, navegação, etc.
    """
    
    def __init__(self, config: ActiveInferenceConfig):
        self.config = config
        
        # Matrizes de transição por tipo de ação
        self.transition_models: Dict[ActionType, np.ndarray] = {}
        self._init_transition_models()
        
        # Modelo de observação
        self.observation_model = self._init_observation_model()
        
        # Prior sobre estados
        self.state_prior_mean = np.zeros(config.state_dim)
        self.state_prior_precision = np.ones(config.state_dim)
        
        # Histórico para aprendizado
        self.transition_history: List[Tuple[np.ndarray, Action, np.ndarray]] = []
        
    def _init_transition_models(self):
        """Inicializa modelos de transição para cada tipo de ação"""
        dim = self.config.state_dim
        
        for action_type in ActionType:
            # Matriz de transição inicial (perto da identidade + ruído)
            A = np.eye(dim) * 0.9 + np.random.randn(dim, dim) * 0.1
            self.transition_models[action_type] = A
    
    def _init_observation_model(self) -> np.ndarray:
        """Inicializa modelo de observação"""
        # C: state_dim → observation_dim (assumindo mesmo tamanho por simplicidade)
        dim = self.config.state_dim
        C = np.eye(dim) + np.random.randn(dim, dim) * 0.1
        return C
    
    def predict_next_state(
        self, 
        current_state: np.ndarray, 
        action: Action
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz próximo estado dado estado atual e ação.
        
        Returns:
            mean: Estado predito
            precision: Confiança na predição
        """
        A = self.transition_models[action.action_type]
        
        # Predição linear
        predicted_mean = A @ current_state
        
        # Incerteza aumenta com ações mais exploratórias
        base_precision = np.ones(self.config.state_dim)
        if action.action_type in [ActionType.EXPLORE_CLUSTER, ActionType.BRIDGE_CONCEPTS]:
            base_precision *= 0.5  # Mais incerteza para exploração
        
        return predicted_mean, base_precision
    
    def predict_observation(
        self, 
        state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz observação dado estado.
        
        Returns:
            mean: Observação predita
            precision: Confiança
        """
        predicted_obs = self.observation_model @ state
        precision = np.ones_like(predicted_obs)  # Simplificado
        return predicted_obs, precision
    
    def update_from_experience(
        self,
        prev_state: np.ndarray,
        action: Action,
        next_state: np.ndarray,
        learning_rate: Optional[float] = None
    ):
        """
        Atualiza modelo de transição baseado em experiência.
        
        Δ A = lr * (s_t+1 - A @ s_t) @ s_t.T
        """
        lr = learning_rate or self.config.learning_rate
        
        A = self.transition_models[action.action_type]
        prediction = A @ prev_state
        error = next_state - prediction
        
        # Gradiente
        dA = lr * np.outer(error, prev_state)
        self.transition_models[action.action_type] += dA
        
        # Registra para análise
        self.transition_history.append((prev_state.copy(), action, next_state.copy()))
        
        # Limita histórico
        if len(self.transition_history) > 1000:
            self.transition_history = self.transition_history[-500:]


# =============================================================================
# AGENTE DE ACTIVE INFERENCE
# =============================================================================

class ActiveInferenceAgent:
    """
    Agente de Active Inference para Alexandria.
    
    O agente:
    1. Mantém beliefs sobre o estado do conhecimento
    2. Tem preferências (goals) sobre estados desejados
    3. Planeja ações para minimizar Expected Free Energy
    4. Executa ações e atualiza beliefs
    
    Ciclo:
        Observe → Update Beliefs → Plan → Act → Repeat
    """
    
    def __init__(
        self,
        config: Optional[ActiveInferenceConfig] = None,
        predictive_coding: Optional[PredictiveCodingNetwork] = None
    ):
        self.config = config or ActiveInferenceConfig()
        self.pc = predictive_coding
        
        # Modelo generativo
        self.generative_model = GenerativeModel(self.config)
        
        # Estado atual
        self.current_state = np.zeros(self.config.state_dim)
        self.state_precision = np.ones(self.config.state_dim)
        
        # Beliefs sobre diferentes conceitos/regiões
        self.beliefs: Dict[str, Belief] = {}
        
        # Preferências
        self.preferences = Preference()
        
        # Histórico de ações
        self.action_history: List[Tuple[int, Action, float]] = []  # (timestep, action, EFE)
        self.timestep = 0
        
        # Cache de ações candidatas
        self.action_queue: List[Action] = []
        
        # Gaps identificados
        self.knowledge_gaps: Dict[str, float] = {}  # concept -> uncertainty
        
    # =========================================================================
    # BELIEF UPDATE
    # =========================================================================
    
    def update_belief(
        self,
        concept_id: str,
        observation: np.ndarray,
        observation_precision: Optional[np.ndarray] = None
    ) -> Belief:
        """
        Atualiza belief sobre um conceito dado nova observação.
        
        Usa regra de Bayes com distribuições Gaussianas:
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * observation) / posterior_precision
        """
        obs_precision = observation_precision if observation_precision is not None else np.ones_like(observation)
        
        if concept_id in self.beliefs:
            belief = self.beliefs[concept_id]
            
            # Bayesian update
            new_precision = belief.precision + obs_precision
            new_mean = (belief.precision * belief.mean + obs_precision * observation) / new_precision
            
            belief.mean = new_mean
            belief.precision = new_precision
            belief.last_updated = self.timestep
            belief.observation_count += 1
        else:
            # Novo conceito
            belief = Belief(
                concept_id=concept_id,
                mean=observation.copy(),
                precision=obs_precision.copy(),
                last_updated=self.timestep,
                observation_count=1
            )
            self.beliefs[concept_id] = belief
        
        return belief
    
    def decay_beliefs(self):
        """
        Aplica decay à precisão dos beliefs.
        
        Conceitos não observados recentemente perdem confiança.
        """
        for belief in self.beliefs.values():
            time_since_update = self.timestep - belief.last_updated
            decay_factor = self.config.belief_decay ** time_since_update
            belief.precision *= decay_factor
            
            # Minimum precision
            belief.precision = np.maximum(belief.precision, 0.1)
    
    def get_uncertain_beliefs(self, top_k: int = 10) -> List[Belief]:
        """Retorna beliefs com maior incerteza"""
        sorted_beliefs = sorted(
            self.beliefs.values(),
            key=lambda b: b.uncertainty,
            reverse=True
        )
        return sorted_beliefs[:top_k]
    
    # =========================================================================
    # ACTION GENERATION
    # =========================================================================
    
    def generate_candidate_actions(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Action]:
        """
        Gera conjunto de ações candidatas baseado no estado atual.
        
        Considera:
        - Beliefs incertos (exploração)
        - Preferências do usuário
        - Gaps no conhecimento
        - Conexões potenciais
        """
        candidates = []
        context = context or {}
        
        # 1. Ações de exploração para beliefs incertos
        uncertain = self.get_uncertain_beliefs(5)
        for belief in uncertain:
            candidates.append(Action(
                action_type=ActionType.EXPLORE_CLUSTER,
                target=belief.concept_id,
                parameters={'uncertainty': belief.uncertainty},
                expected_information_gain=belief.uncertainty
            ))
        
        # 2. Ações de aprofundamento para beliefs frequentes
        frequent = sorted(
            self.beliefs.values(),
            key=lambda b: b.observation_count,
            reverse=True
        )[:3]
        for belief in frequent:
            candidates.append(Action(
                action_type=ActionType.DEEPEN_TOPIC,
                target=belief.concept_id,
                parameters={'observations': belief.observation_count}
            ))
        
        # 3. Ações de bridging entre conceitos distantes
        if len(self.beliefs) >= 2:
            concepts = list(self.beliefs.keys())
            for i in range(min(3, len(concepts))):
                for j in range(i + 1, min(4, len(concepts))):
                    c1, c2 = concepts[i], concepts[j]
                    candidates.append(Action(
                        action_type=ActionType.BRIDGE_CONCEPTS,
                        target=f"{c1}+{c2}",
                        parameters={'concept1': c1, 'concept2': c2}
                    ))
        
        # 4. Ações para preencher gaps
        for gap_concept, gap_uncertainty in self.knowledge_gaps.items():
            candidates.append(Action(
                action_type=ActionType.FILL_GAP,
                target=gap_concept,
                parameters={'gap_severity': gap_uncertainty},
                expected_information_gain=gap_uncertainty * 1.5  # Bonus para gaps
            ))
        
        # 5. Ações baseadas em preferências
        for topic, weight in self.preferences.topic_weights.items():
            if weight > 0.5:
                candidates.append(Action(
                    action_type=ActionType.QUERY_SEARCH,
                    target=topic,
                    parameters={'preference_weight': weight}
                ))
        
        # 6. Sempre inclui opção de consolidar/descansar
        candidates.append(Action(
            action_type=ActionType.CONSOLIDATE,
            target="all",
            parameters={}
        ))
        
        # 7. Ações do contexto (se fornecido)
        if 'recent_queries' in context:
            for query in context['recent_queries'][:3]:
                candidates.append(Action(
                    action_type=ActionType.FOLLOW_CONNECTION,
                    target=query,
                    parameters={'source': 'recent_query'}
                ))
        
        return candidates
    
    # =========================================================================
    # EXPECTED FREE ENERGY
    # =========================================================================
    
    def compute_expected_free_energy(
        self,
        action: Action,
        current_state: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computa Expected Free Energy (EFE) para uma ação.
        
        G(π) = Risk + Ambiguity
        
        Risk: Quão longe do estado preferido a ação nos leva
        Ambiguity: Quanta incerteza sobre o outcome
        
        Menor G = melhor ação
        
        Returns:
            G: Expected Free Energy (menor é melhor)
            components: Breakdown dos termos
        """
        state = current_state if current_state is not None else self.current_state
        
        # Prediz próximo estado
        pred_state, pred_precision = self.generative_model.predict_next_state(state, action)
        
        # Prediz observação
        pred_obs, obs_precision = self.generative_model.predict_observation(pred_state)
        
        # === RISK ===
        # Divergência do estado preferido
        preferred_state = self._get_preferred_state(action)
        risk = np.sum((pred_state - preferred_state) ** 2 * pred_precision)
        
        # === AMBIGUITY ===
        # Entropia esperada das observações
        # H(o|s,π) ≈ -log(precision)
        ambiguity = -np.sum(np.log(obs_precision + 0.01))
        
        # === INFORMATION GAIN (bonus negativo) ===
        # Quanto de incerteza a ação reduz
        info_gain = action.expected_information_gain
        
        # === NOVELTY BONUS ===
        # Bonus para ações que exploram
        novelty = 0
        if action.action_type in [ActionType.EXPLORE_CLUSTER, ActionType.BRIDGE_CONCEPTS]:
            novelty = self.config.novelty_bonus * self.preferences.novelty_preference
        
        # === EFE TOTAL ===
        G = (
            self.config.risk_weight * risk +
            self.config.ambiguity_weight * ambiguity -
            info_gain -
            novelty
        )
        
        components = {
            'risk': risk,
            'ambiguity': ambiguity,
            'information_gain': info_gain,
            'novelty': novelty,
            'total_G': G
        }
        
        return G, components
    
    def _get_preferred_state(self, action: Action) -> np.ndarray:
        """
        Retorna estado preferido baseado na ação e preferências.
        
        Em implementação completa, isso seria aprendido ou especificado.
        """
        # Default: estado atual com pequena modificação na direção da ação
        preferred = self.current_state.copy()
        
        # Modifica baseado no tipo de ação
        if action.action_type == ActionType.DEEPEN_TOPIC:
            # Preferência por estados similares ao atual (exploitar)
            pass
        elif action.action_type == ActionType.EXPLORE_CLUSTER:
            # Preferência por estados diferentes (explorar)
            preferred += np.random.randn(len(preferred)) * 0.5
        elif action.action_type == ActionType.BRIDGE_CONCEPTS:
            # Preferência por estados intermediários
            if action.target in self.beliefs:
                belief = self.beliefs[action.target]
                preferred = (preferred + belief.mean) / 2
        
        return preferred
    
    # =========================================================================
    # ACTION SELECTION
    # =========================================================================
    
    def select_action(
        self,
        candidates: Optional[List[Action]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Seleciona melhor ação via softmax sobre negative EFE.
        
        P(a) ∝ exp(-G(a) / temperature)
        
        Returns:
            action: Ação selecionada
            info: Informações sobre a seleção
        """
        if candidates is None:
            candidates = self.generate_candidate_actions(context)
        
        if not candidates:
            # Fallback: descansar
            return Action(action_type=ActionType.REST, target="none"), {'reason': 'no_candidates'}
        
        # Computa EFE para cada candidato
        efes = []
        components_list = []
        for action in candidates:
            G, components = self.compute_expected_free_energy(action)
            efes.append(G)
            components_list.append(components)
            action.priority = -G  # Maior priority = menor G
        
        # Softmax selection
        efes = np.array(efes)
        
        # NaN handling: replace with infinity (lowest probability)
        efes = np.nan_to_num(efes, nan=np.inf)
        
        # Normaliza para estabilidade numérica
        # Subtrai min em vez de max pois estamos minimizando EFE (note o sinal negativo no expoente)
        # Prob = exp(-EFE)
        min_efe = np.min(efes)
        if np.isinf(min_efe):
            # Fallback if all are inf
            probs = np.ones(len(efes)) / len(efes)
        else:
            efes_shifted = efes - min_efe
            probs = np.exp(-efes_shifted / self.config.temperature)
            
            # Safe normalization
            params_sum = np.sum(probs)
            if params_sum == 0 or np.isnan(params_sum):
                probs = np.ones(len(efes)) / len(efes)
            else:
                probs = probs / params_sum
        
        # Amostra ou pega argmax
        if self.config.temperature > 0.1:
            selected_idx = np.random.choice(len(candidates), p=probs)
        else:
            selected_idx = np.argmin(efes)
        
        selected_action = candidates[selected_idx]
        selected_action.expected_information_gain = components_list[selected_idx]['information_gain']
        selected_action.expected_risk = components_list[selected_idx]['risk']
        
        # Registra
        self.action_history.append((self.timestep, selected_action, efes[selected_idx]))
        
        return selected_action, {
            'all_candidates': len(candidates),
            'selected_idx': selected_idx,
            'selected_EFE': efes[selected_idx],
            'selection_prob': probs[selected_idx],
            'components': components_list[selected_idx],
            'top_3': [(candidates[i].action_type.name, -efes[i]) for i in np.argsort(efes)[:3]]
        }
    
    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================
    
    def execute_action(
        self,
        action: Action,
        alexandria_interface: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executa ação e retorna resultado.
        
        Em integração real com Alexandria:
        - QUERY_SEARCH: Executa busca no LanceDB
        - EXPLORE_CLUSTER: Ativa propagação mycelial
        - etc.
        
        Aqui simulamos os efeitos.
        """
        result = {
            'action': action,
            'success': True,
            'observations': [],
            'new_beliefs': [],
            'state_change': None
        }
        
        if alexandria_interface is not None:
            # Execução real (quando integrado)
            result = self._execute_with_alexandria(action, alexandria_interface)
        else:
            # Simulação
            result = self._simulate_action(action)
        
        # Atualiza estado
        if result['state_change'] is not None:
            prev_state = self.current_state.copy()
            self.current_state = result['state_change']
            
            # Atualiza modelo generativo
            self.generative_model.update_from_experience(
                prev_state, action, self.current_state
            )
        
        self.timestep += 1
        self.decay_beliefs()
        
        return result
    
    def _simulate_action(self, action: Action) -> Dict[str, Any]:
        """Simula execução de ação (para testes)"""
        result = {
            'action': action,
            'success': True,
            'observations': [],
            'new_beliefs': [],
            'state_change': None
        }
        
        # Simula transição de estado
        pred_state, _ = self.generative_model.predict_next_state(
            self.current_state, action
        )
        
        # Adiciona ruído (simulando incerteza do mundo)
        noise = np.random.randn(len(pred_state)) * 0.1
        new_state = pred_state + noise
        
        result['state_change'] = new_state
        
        # Simula observações
        num_obs = {
            ActionType.QUERY_SEARCH: 5,
            ActionType.EXPLORE_CLUSTER: 3,
            ActionType.FOLLOW_CONNECTION: 2,
            ActionType.DEEPEN_TOPIC: 4,
            ActionType.BRIDGE_CONCEPTS: 2,
            ActionType.FILL_GAP: 3,
            ActionType.CONSOLIDATE: 0,
            ActionType.REST: 0
        }.get(action.action_type, 1)
        
        for i in range(num_obs):
            obs = new_state + np.random.randn(len(new_state)) * 0.2
            result['observations'].append(obs)
            
            # Cria belief para observação
            concept_id = f"{action.target}_{i}"
            belief = self.update_belief(concept_id, obs)
            result['new_beliefs'].append(belief.concept_id)
        
        return result
    
    def _execute_with_alexandria(
        self,
        action: Action,
        interface: Any
    ) -> Dict[str, Any]:
        """
        Execução real com sistema Alexandria.
        
        Interface esperada:
        - interface.search(query) → List[Document]
        - interface.propagate(indices, steps) → List[int]
        - interface.get_embedding(text) → np.ndarray
        """
        result = {
            'action': action,
            'success': True,
            'observations': [],
            'new_beliefs': [],
            'state_change': None
        }
        
        try:
            if action.action_type == ActionType.QUERY_SEARCH:
                # Busca por query
                if hasattr(interface, 'search'):
                    docs = interface.search(action.target, limit=5)
                    for doc in docs:
                        if hasattr(interface, 'get_embedding'):
                            obs = interface.get_embedding(doc.get('text', ''))
                            result['observations'].append(obs)
                            
            elif action.action_type == ActionType.EXPLORE_CLUSTER:
                # Propagação mycelial
                if hasattr(interface, 'propagate'):
                    # Assume que target é um índice ou conjunto de índices
                    indices = action.parameters.get('indices', [])
                    expanded = interface.propagate(indices, steps=3)
                    result['expanded_indices'] = expanded
                    
            elif action.action_type == ActionType.FOLLOW_CONNECTION:
                # Segue conexão específica
                if hasattr(interface, 'get_connections'):
                    connections = interface.get_connections(action.target)
                    result['connections'] = connections
                    
            # Atualiza estado baseado em observações
            if result['observations']:
                new_state = np.mean(result['observations'], axis=0)
                result['state_change'] = new_state
                
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    # =========================================================================
    # PLANNING
    # =========================================================================
    
    def plan(
        self,
        horizon: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Action, float]]:
        """
        Planeja sequência de ações para horizonte futuro.
        
        Usa tree search simplificado com EFE como heurística.
        
        Returns:
            plan: Lista de (ação, EFE esperado)
        """
        horizon = horizon or self.config.planning_horizon
        plan = []
        
        # Estado hipotético para simulação
        sim_state = self.current_state.copy()
        
        for step in range(horizon):
            # Gera candidatos
            candidates = self.generate_candidate_actions(context)
            
            # Avalia cada um
            best_action = None
            best_efe = float('inf')
            
            for action in candidates:
                efe, _ = self.compute_expected_free_energy(action, sim_state)
                if efe < best_efe:
                    best_efe = efe
                    best_action = action
            
            if best_action is None:
                break
                
            plan.append((best_action, best_efe))
            
            # Simula transição
            sim_state, _ = self.generative_model.predict_next_state(sim_state, best_action)
        
        return plan
    
    # =========================================================================
    # GAP DETECTION
    # =========================================================================
    
    def detect_knowledge_gaps(
        self,
        min_uncertainty: float = 0.5
    ) -> Dict[str, float]:
        """
        Detecta gaps no conhecimento baseado em:
        1. Beliefs com alta incerteza
        2. Regiões do espaço não exploradas
        3. Conexões faltantes entre conceitos
        """
        gaps = {}
        
        # 1. Beliefs incertos
        for concept_id, belief in self.beliefs.items():
            if belief.uncertainty > min_uncertainty:
                gaps[concept_id] = belief.uncertainty
        
        # 2. Conceitos mencionados mas não observados
        # (seria preenchido com dados reais do Alexandria)
        
        # 3. Pontes faltantes entre clusters distantes
        if len(self.beliefs) >= 2:
            concepts = list(self.beliefs.keys())
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    b1, b2 = self.beliefs[c1], self.beliefs[c2]
                    # Distância entre beliefs
                    dist = np.linalg.norm(b1.mean - b2.mean)
                    if dist > 2.0:  # Threshold
                        gap_id = f"bridge:{c1}:{c2}"
                        gaps[gap_id] = dist / 10.0
        
        self.knowledge_gaps = gaps
        return gaps
    
    # =========================================================================
    # ANALYSIS & DIAGNOSTICS
    # =========================================================================
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Retorna estado completo do agente para análise"""
        return {
            'timestep': self.timestep,
            'current_state_norm': float(np.linalg.norm(self.current_state)),
            'state_precision_mean': float(np.mean(self.state_precision)),
            'num_beliefs': len(self.beliefs),
            'beliefs_summary': {
                cid: {
                    'uncertainty': b.uncertainty,
                    'observations': b.observation_count,
                    'age': self.timestep - b.last_updated
                }
                for cid, b in list(self.beliefs.items())[:10]
            },
            'num_knowledge_gaps': len(self.knowledge_gaps),
            'top_gaps': dict(sorted(
                self.knowledge_gaps.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'action_history_length': len(self.action_history),
            'recent_actions': [
                (ts, a.action_type.name, efe) 
                for ts, a, efe in self.action_history[-5:]
            ],
            'preferences': {
                'novelty': self.preferences.novelty_preference,
                'depth': self.preferences.depth_preference,
                'topics': self.preferences.topic_weights
            }
        }
    
    def get_recommended_action(self) -> Dict[str, Any]:
        """
        Retorna recomendação de próxima ação com explicação.
        """
        # Detecta gaps
        gaps = self.detect_knowledge_gaps()
        
        # Gera candidatos
        candidates = self.generate_candidate_actions()
        
        # Seleciona
        action, info = self.select_action(candidates)
        
        # Gera explicação
        explanation = self._explain_action(action, info)
        
        return {
            'action': {
                'type': action.action_type.name,
                'target': action.target,
                'parameters': action.parameters
            },
            'expected_free_energy': info['selected_EFE'],
            'confidence': info['selection_prob'],
            'explanation': explanation,
            'alternatives': info['top_3'],
            'gaps_detected': len(gaps)
        }
    
    def _explain_action(self, action: Action, info: Dict) -> str:
        """Gera explicação em linguagem natural para a ação"""
        components = info.get('components', {})
        
        explanations = {
            ActionType.QUERY_SEARCH: f"Buscar por '{action.target}' para expandir conhecimento",
            ActionType.EXPLORE_CLUSTER: f"Explorar cluster '{action.target}' - alta incerteza detectada",
            ActionType.FOLLOW_CONNECTION: f"Seguir conexão mycelial a partir de '{action.target}'",
            ActionType.DEEPEN_TOPIC: f"Aprofundar em '{action.target}' - tópico frequentemente observado",
            ActionType.BRIDGE_CONCEPTS: f"Buscar conexão entre conceitos: {action.target}",
            ActionType.FILL_GAP: f"Preencher gap de conhecimento: '{action.target}'",
            ActionType.CONSOLIDATE: "Consolidar conhecimento existente antes de explorar mais",
            ActionType.REST: "Pausar e permitir settling dos beliefs"
        }
        
        base = explanations.get(action.action_type, f"Executar {action.action_type.name}")
        
        # Adiciona detalhes sobre componentes
        if components:
            risk = components.get('risk', 0)
            info_gain = components.get('information_gain', 0)
            
            if info_gain > 0.5:
                base += f" (alto ganho de informação esperado: {info_gain:.2f})"
            if risk < 0.3:
                base += " (baixo risco)"
        
        return base
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Salva estado completo do agente"""
        path = path or self.config.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': {
                'state_dim': self.config.state_dim,
                'planning_horizon': self.config.planning_horizon,
                'temperature': self.config.temperature
            },
            'current_state': self.current_state,
            'state_precision': self.state_precision,
            'beliefs': {
                cid: {
                    'mean': b.mean,
                    'precision': b.precision,
                    'last_updated': b.last_updated,
                    'observation_count': b.observation_count
                }
                for cid, b in self.beliefs.items()
            },
            'preferences': {
                'topic_weights': self.preferences.topic_weights,
                'novelty_preference': self.preferences.novelty_preference,
                'depth_preference': self.preferences.depth_preference
            },
            'knowledge_gaps': self.knowledge_gaps,
            'timestep': self.timestep,
            'transition_models': {
                at.name: A for at, A in self.generative_model.transition_models.items()
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        return path
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Carrega estado salvo"""
        path = path or self.config.save_path
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.current_state = state['current_state']
            self.state_precision = state['state_precision']
            self.timestep = state.get('timestep', 0)
            self.knowledge_gaps = state.get('knowledge_gaps', {})
            
            # Restaura beliefs
            for cid, bdata in state.get('beliefs', {}).items():
                self.beliefs[cid] = Belief(
                    concept_id=cid,
                    mean=bdata['mean'],
                    precision=bdata['precision'],
                    last_updated=bdata['last_updated'],
                    observation_count=bdata['observation_count']
                )
            
            # Restaura preferências
            prefs = state.get('preferences', {})
            self.preferences.topic_weights = prefs.get('topic_weights', {})
            self.preferences.novelty_preference = prefs.get('novelty_preference', 0.5)
            self.preferences.depth_preference = prefs.get('depth_preference', 0.5)
            
            # Restaura modelos de transição
            for at_name, A in state.get('transition_models', {}).items():
                at = ActionType[at_name]
                self.generative_model.transition_models[at] = A
            
            return True
        except Exception as e:
            print(f"Erro ao carregar Active Inference: {e}")
            return False


# =============================================================================
# INTEGRAÇÃO COM ALEXANDRIA
# =============================================================================

class AlexandriaInterface:
    """
    Interface entre Active Inference e sistema Alexandria.
    
    Traduz ações do agente em operações concretas do Alexandria.
    """
    
    def __init__(
        self,
        semantic_memory=None,    # LanceDB
        mycelial_network=None,   # MycelialReasoning
        vqvae=None,              # MONOLITH
        embedding_model=None     # sentence-transformers
    ):
        self.semantic_memory = semantic_memory
        self.mycelial = mycelial_network
        self.vqvae = vqvae
        self.embedding_model = embedding_model
        
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Busca semântica via LanceDB"""
        if self.semantic_memory is None:
            return []
        
        # Implementação real dependeria do setup específico
        # Aqui está o pattern esperado:
        #
        # embedding = self.embedding_model.encode(query)
        # results = self.semantic_memory.search(embedding).limit(limit).to_list()
        # return results
        
        return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para texto"""
        if self.embedding_model is None:
            return np.random.randn(384)  # Fallback
        
        # return self.embedding_model.encode(text)
        return np.random.randn(384)
    
    def propagate(self, indices: List[int], steps: int = 3) -> List[int]:
        """Propagação mycelial"""
        if self.mycelial is None:
            return indices
        
        # return self.mycelial.propagate(indices, steps)
        return indices
    
    def get_connections(self, concept: str) -> List[Dict]:
        """Retorna conexões de um conceito"""
        if self.mycelial is None:
            return []
        
        # return self.mycelial.get_connections(concept)
        return []


class ActiveInferenceAlexandria:
    """
    Sistema completo de Active Inference para Alexandria.
    
    Combina:
    - Agente de Active Inference
    - Interface com Alexandria
    - Ciclo de percepção-ação
    """
    
    def __init__(
        self,
        agent: Optional[ActiveInferenceAgent] = None,
        interface: Optional[AlexandriaInterface] = None,
        predictive_coding: Optional[PredictiveCodingNetwork] = None
    ):
        self.agent = agent or ActiveInferenceAgent(
            predictive_coding=predictive_coding
        )
        self.interface = interface or AlexandriaInterface()
        self.pc = predictive_coding
        
        # Ciclo ativo
        self.is_running = False
        self.cycle_count = 0
        
    def perception_action_cycle(
        self,
        external_observation: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa um ciclo completo de percepção-ação.
        
        1. Observa (processa input externo se houver)
        2. Atualiza beliefs
        3. Detecta gaps
        4. Seleciona ação
        5. Executa ação
        6. Atualiza estado
        
        Returns:
            result: Resultado do ciclo
        """
        result = {
            'cycle': self.cycle_count,
            'observation_processed': False,
            'action_taken': None,
            'state_updated': False,
            'gaps_detected': 0
        }
        
        # 1. Processa observação externa
        if external_observation is not None:
            # Se temos Predictive Coding, usa para processar
            if self.pc is not None:
                pc_result = self.pc.learn_from_input(external_observation)
                processed_obs = pc_result['code']
            else:
                processed_obs = external_observation
            
            # Atualiza belief genérico
            self.agent.update_belief(f"obs_{self.cycle_count}", processed_obs)
            result['observation_processed'] = True
        
        # 2. Detecta gaps
        gaps = self.agent.detect_knowledge_gaps()
        result['gaps_detected'] = len(gaps)
        
        # 3. Seleciona e executa ação
        action, selection_info = self.agent.select_action(context=context)
        execution_result = self.agent.execute_action(action, self.interface)
        
        result['action_taken'] = {
            'type': action.action_type.name,
            'target': action.target,
            'success': execution_result['success'],
            'efe': selection_info['selected_EFE']
        }
        result['state_updated'] = execution_result['state_change'] is not None
        
        self.cycle_count += 1
        
        return result
    
    def run_cycles(
        self,
        num_cycles: int,
        callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Executa múltiplos ciclos de percepção-ação.
        """
        results = []
        
        for i in range(num_cycles):
            result = self.perception_action_cycle()
            results.append(result)
            
            if callback:
                callback(result)
        
        return results
    
    def suggest_exploration(self) -> Dict[str, Any]:
        """
        Sugere direção de exploração baseado no estado atual.
        
        Útil para:
        - Sugerir queries ao usuário
        - Identificar tópicos para aprofundar
        - Mostrar gaps no conhecimento
        """
        recommendation = self.agent.get_recommended_action()
        gaps = self.agent.knowledge_gaps
        
        return {
            'recommended_action': recommendation,
            'knowledge_gaps': gaps,
            'agent_state': self.agent.get_agent_state(),
            'suggested_queries': self._generate_query_suggestions()
        }
    
    def _generate_query_suggestions(self, num_suggestions: int = 5) -> List[str]:
        """Gera sugestões de queries baseado nos gaps e beliefs"""
        suggestions = []
        
        # Baseado em gaps
        for gap_id, uncertainty in sorted(
            self.agent.knowledge_gaps.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            if gap_id.startswith('bridge:'):
                parts = gap_id.split(':')
                if len(parts) >= 3:
                    suggestions.append(f"connection between {parts[1]} and {parts[2]}")
            else:
                suggestions.append(gap_id)
        
        # Baseado em beliefs incertos
        uncertain = self.agent.get_uncertain_beliefs(2)
        for belief in uncertain:
            suggestions.append(f"more about {belief.concept_id}")
        
        return suggestions[:num_suggestions]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Status completo do sistema"""
        return {
            'agent': self.agent.get_agent_state(),
            'has_predictive_coding': self.pc is not None,
            'has_interface': self.interface is not None,
            'cycle_count': self.cycle_count,
            'is_running': self.is_running
        }


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def create_active_inference_system(
    state_dim: int = 64,
    load_existing: bool = True,
    use_predictive_coding: bool = True
) -> ActiveInferenceAlexandria:
    """
    Factory function para criar sistema completo de Active Inference.
    """
    # Cria Predictive Coding se solicitado
    pc = None
    if use_predictive_coding:
        try:
            pc = create_predictive_coding_system(
                code_dim=state_dim,
                load_existing=load_existing
            )
        except:
            print("⚠️ Predictive Coding não disponível, continuando sem")
    
    # Cria config
    config = ActiveInferenceConfig(state_dim=state_dim)
    
    # Cria agente
    agent = ActiveInferenceAgent(config=config, predictive_coding=pc)
    
    # Tenta carregar estado existente
    if load_existing:
        loaded = agent.load_state()
        if loaded:
            print(f"✅ Active Inference carregado: timestep {agent.timestep}")
        else:
            print("🌱 Active Inference inicializado fresh")
    
    # Cria sistema completo
    system = ActiveInferenceAlexandria(agent=agent, predictive_coding=pc)
    
    return system


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ACTIVE INFERENCE - ALEXANDRIA")
    print("=" * 70)
    
    # Criar sistema
    system = create_active_inference_system(
        state_dim=64,
        load_existing=False,
        use_predictive_coding=False  # Simplifica teste
    )
    
    agent = system.agent
    
    # Simular algumas observações
    print("\n🔄 SIMULANDO CICLOS DE PERCEPÇÃO-AÇÃO...")
    
    for i in range(10):
        # Observação simulada
        obs = np.random.randn(64)
        
        # Executa ciclo
        result = system.perception_action_cycle(external_observation=obs)
        
        if i % 3 == 0:
            print(f"\n   Ciclo {result['cycle']}:")
            print(f"      Ação: {result['action_taken']['type']}")
            print(f"      Target: {result['action_taken']['target'][:30]}...")
            print(f"      EFE: {result['action_taken']['efe']:.3f}")
            print(f"      Gaps: {result['gaps_detected']}")
    
    # Análise do estado
    print("\n📊 ESTADO DO AGENTE:")
    state = agent.get_agent_state()
    print(f"   Timestep: {state['timestep']}")
    print(f"   Beliefs: {state['num_beliefs']}")
    print(f"   Knowledge gaps: {state['num_knowledge_gaps']}")
    print(f"   Preferência novelty: {state['preferences']['novelty']}")
    
    # Recomendação
    print("\n🎯 RECOMENDAÇÃO:")
    rec = agent.get_recommended_action()
    print(f"   Ação: {rec['action']['type']}")
    print(f"   Target: {rec['action']['target']}")
    print(f"   Explicação: {rec['explanation']}")
    print(f"   Confiança: {rec['confidence']:.2%}")
    
    # Top gaps
    print("\n🕳️ GAPS DE CONHECIMENTO:")
    for gap_id, uncertainty in list(state['top_gaps'].items())[:5]:
        print(f"   {gap_id}: {uncertainty:.3f}")
    
    # Sugestões de exploração
    print("\n💡 SUGESTÕES DE EXPLORAÇÃO:")
    exploration = system.suggest_exploration()
    for i, query in enumerate(exploration['suggested_queries'], 1):
        print(f"   {i}. {query}")
    
    # Planning
    print("\n📋 PLANO DE AÇÕES (próximos 5 passos):")
    plan = agent.plan(horizon=5)
    for i, (action, efe) in enumerate(plan, 1):
        print(f"   {i}. {action.action_type.name}: {action.target[:25]}... (EFE: {efe:.3f})")
    
    # Salvar
    save_path = agent.save_state()
    print(f"\n💾 Estado salvo em: {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ ACTIVE INFERENCE PRONTO PARA INTEGRAÇÃO")
    print("=" * 70)
    
    print("""
    
ARQUITETURA COMPLETA:
=====================

    Observação (embedding)
           ↓
    ┌──────────────────────────────────────────────┐
    │            ACTIVE INFERENCE AGENT            │
    │                                              │
    │   ┌────────────┐    ┌────────────────────┐  │
    │   │  Beliefs   │←──→│  Generative Model  │  │
    │   │  Q(s|o)    │    │  P(o,s|π)          │  │
    │   └────────────┘    └────────────────────┘  │
    │          ↓                    ↓              │
    │   ┌────────────┐    ┌────────────────────┐  │
    │   │Preferences │    │ Expected Free      │  │
    │   │  P(o)      │───→│ Energy G(π)        │  │
    │   └────────────┘    └────────────────────┘  │
    │                            ↓                │
    │                   ┌────────────────┐        │
    │                   │ Action Select  │        │
    │                   │ argmin G(π)    │        │
    │                   └────────────────┘        │
    └──────────────────────────────────────────────┘
           ↓
    Ação → Alexandria (search, propagate, etc.)


TIPOS DE AÇÃO:
==============

    QUERY_SEARCH      → Buscar papers específicos
    EXPLORE_CLUSTER   → Explorar região incerta
    FOLLOW_CONNECTION → Navegar rede mycelial
    DEEPEN_TOPIC      → Aprofundar conhecimento
    BRIDGE_CONCEPTS   → Conectar conceitos distantes
    FILL_GAP          → Preencher lacuna identificada
    CONSOLIDATE       → Consolidar antes de expandir
    REST              → Pausar para settling


CAMINHO DE EVOLUÇÃO:
====================

    ✅ Hebbian (base)
    ✅ Meta-Hebbian (regras aprendidas)
    ✅ Predictive Coding (propaga erros)
    ✅ Active Inference (age no mundo) ← VOCÊ ESTÁ AQUI
    ⬜ Free Energy completo (futuro)


INTEGRAÇÃO COM ALEXANDRIA:
===========================

    from active_inference import create_active_inference_system
    
    # Criar sistema
    system = create_active_inference_system()
    
    # Processar observação
    embedding = model.encode("vector quantization neural networks")
    result = system.perception_action_cycle(external_observation=embedding)
    
    # Ver recomendação
    rec = system.agent.get_recommended_action()
    print(f"Sugestão: {rec['action']['type']} - {rec['explanation']}")
    
    # Rodar múltiplos ciclos
    results = system.run_cycles(20)
    
    # Ver sugestões de exploração
    exploration = system.suggest_exploration()
    for query in exploration['suggested_queries']:
        print(f"Buscar: {query}")
    
    """)
