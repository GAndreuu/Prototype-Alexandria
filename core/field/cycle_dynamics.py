"""
CycleDynamics: Expansão → Configuração → Compressão
===================================================

Este é o coração do sistema - o ciclo que faz tudo evoluir.

O ciclo não é um loop simples. Cada iteração transforma as próprias
regras de transformação. É auto-modificante.

Fases:
1. EXPANSÃO: Espaço cresce em dimensões que não existiam
2. CONFIGURAÇÃO: Elementos se arranjam em padrões "ilógico-lógicos"
3. COMPRESSÃO: Colapsa redefinindo a própria métrica

A "mágica" está na meta-regra: o ciclo modifica como ciclos futuros
vão funcionar. Isso é o que permite aprendizado real.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .manifold import DynamicManifold, ManifoldPoint
from .metric import RiemannianMetric
from .free_energy_field import FreeEnergyField, FieldState
from .geodesic_flow import GeodesicFlow, GeodesicPath


logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Fases do ciclo."""
    IDLE = "idle"
    EXPANSION = "expansion"
    CONFIGURATION = "configuration"
    COMPRESSION = "compression"
    META_UPDATE = "meta_update"


@dataclass
class CycleConfig:
    """Configuração do ciclo dinâmico."""
    # Expansão
    expansion_threshold: float = 0.7      # F médio acima disso -> expande
    max_expansion_dims: int = 32          # Máximo de dimensões a adicionar por ciclo
    expansion_rate: float = 0.1           # Fração de dims a adicionar
    
    # Configuração
    configuration_steps: int = 50         # Passos de relaxação
    configuration_temperature: float = 1.0 # T durante configuração
    configuration_cooling_rate: float = 0.95
    
    # Compressão
    compression_threshold: float = 0.3    # F médio abaixo disso -> comprime
    min_compression_dims: int = 0         # Mínimo de dims extras a manter
    compression_rate: float = 0.1         # Fração de dims a remover
    
    # Meta-aprendizado
    meta_learning_rate: float = 0.01      # Taxa de atualização das regras
    history_length: int = 10              # Ciclos passados a considerar


@dataclass
class TransitionRule:
    """Regra de transição que pode evoluir."""
    expansion_weights: np.ndarray = None   # Pesos para decidir onde expandir
    compression_weights: np.ndarray = None # Pesos para decidir o que comprimir
    configuration_bias: np.ndarray = None  # Bias durante configuração
    
    # Histórico para meta-aprendizado
    history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.expansion_weights is None:
            self.expansion_weights = np.ones(10)  # Placeholder
        if self.compression_weights is None:
            self.compression_weights = np.ones(10)
        if self.configuration_bias is None:
            self.configuration_bias = np.zeros(10)


@dataclass
class CycleState:
    """Estado de um ciclo completo."""
    phase: CyclePhase
    cycle_number: int
    
    # Estados do campo em cada fase
    pre_expansion: Optional[FieldState] = None
    post_expansion: Optional[FieldState] = None
    post_configuration: Optional[FieldState] = None
    post_compression: Optional[FieldState] = None
    
    # Métricas
    dimensions_added: int = 0
    dimensions_removed: int = 0
    free_energy_delta: float = 0.0
    attractors_formed: int = 0
    
    # Meta
    rule_delta: Optional[Dict] = None


class CycleDynamics:
    """
    Motor do ciclo Expansão → Configuração → Compressão.
    
    Este componente orquestra a evolução do campo pré-estrutural
    através de ciclos que se auto-modificam.
    
    Attributes:
        manifold: Variedade dinâmica
        metric: Métrica Riemanniana
        field: Campo de energia livre
        flow: Motor geodésico
        config: Configuração
        rule: Regra de transição (evolui)
        cycle_count: Contador de ciclos
    """
    
    def __init__(self,
                 manifold: DynamicManifold,
                 metric: RiemannianMetric,
                 field: FreeEnergyField,
                 flow: GeodesicFlow,
                 config: Optional[CycleConfig] = None):
        self.manifold = manifold
        self.metric = metric
        self.field = field
        self.flow = flow
        self.config = config or CycleConfig()
        
        # Regra de transição (evolui com o tempo)
        self.rule = TransitionRule()
        
        # Estado
        self.current_phase = CyclePhase.IDLE
        self.cycle_count = 0
        self.history: List[CycleState] = []
        
    # =========================================================================
    # CICLO COMPLETO
    # =========================================================================
    
    def run_cycle(self, trigger: Optional[np.ndarray] = None) -> CycleState:
        """
        Executa um ciclo completo: Expansão → Configuração → Compressão.
        
        Args:
            trigger: Embedding opcional que inicia o ciclo
            
        Returns:
            CycleState com resultados do ciclo
        """
        self.cycle_count += 1
        state = CycleState(
            phase=CyclePhase.IDLE,
            cycle_number=self.cycle_count
        )
        
        logger.info(f"Iniciando ciclo {self.cycle_count}")
        
        # Trigger opcional
        if trigger is not None:
            point = self.manifold.embed(trigger)
            self.manifold.add_point(f"trigger_{self.cycle_count}", point)
            self.manifold.activate_point(f"trigger_{self.cycle_count}", intensity=1.0)
            self.metric.deform_at_point(point)
        
        # Estado inicial
        state.pre_expansion = self.field.compute_field()
        
        # FASE 1: EXPANSÃO
        state.phase = CyclePhase.EXPANSION
        dims_added = self._expansion_phase(state)
        state.dimensions_added = dims_added
        state.post_expansion = self.field.compute_field()
        
        # FASE 2: CONFIGURAÇÃO
        state.phase = CyclePhase.CONFIGURATION
        self._configuration_phase(state)
        state.post_configuration = self.field.compute_field()
        
        # FASE 3: COMPRESSÃO
        state.phase = CyclePhase.COMPRESSION
        dims_removed = self._compression_phase(state)
        state.dimensions_removed = dims_removed
        state.post_compression = self.field.compute_field()
        
        # FASE 4: META-ATUALIZAÇÃO
        state.phase = CyclePhase.META_UPDATE
        rule_delta = self._meta_update(state)
        state.rule_delta = rule_delta
        
        # Métricas finais
        if state.pre_expansion and state.post_compression:
            state.free_energy_delta = (
                state.post_compression.mean_free_energy - 
                state.pre_expansion.mean_free_energy
            )
            state.attractors_formed = state.post_compression.num_attractors
            
        # Armazena histórico
        state.phase = CyclePhase.IDLE
        self.history.append(state)
        
        logger.info(f"Ciclo {self.cycle_count} completo. "
                   f"ΔF={state.free_energy_delta:.4f}, "
                   f"dims: +{dims_added}/-{dims_removed}, "
                   f"attractors: {state.attractors_formed}")
        
        return state
    
    # =========================================================================
    # FASE 1: EXPANSÃO
    # =========================================================================
    
    def _expansion_phase(self, state: CycleState) -> int:
        """
        Fase de expansão: espaço cresce em dimensões novas.
        
        Expansão acontece quando energia livre média está alta,
        indicando que o espaço atual não consegue representar
        bem o conhecimento.
        
        Returns:
            Número de dimensões adicionadas
        """
        logger.debug("Entrando fase de EXPANSÃO")
        
        # Decide se precisa expandir
        current_F = self.field.stats()['mean_F']
        
        if current_F < self.config.expansion_threshold:
            logger.debug(f"F={current_F:.4f} < threshold, skip expansion")
            return 0
            
        # Quantas dimensões adicionar
        current_extra = self.manifold.current_dim - self.manifold.config.base_dim
        max_to_add = self.config.max_expansion_dims - current_extra
        
        if max_to_add <= 0:
            logger.debug("Já no máximo de expansão")
            return 0
            
        # Calcula direções de expansão
        n_dims = max(1, int(max_to_add * self.config.expansion_rate))
        n_dims = min(n_dims, max_to_add)
        
        # Direções baseadas em onde a energia é alta
        # TODO: Usar PCA dos gradientes para encontrar direções úteis
        expansion_basis = self._compute_expansion_directions(n_dims)
        
        # Expande
        self.manifold.expand_dimension(n_dims, basis=expansion_basis)
        
        logger.debug(f"Expandido {n_dims} dimensões. "
                    f"Nova dim: {self.manifold.current_dim}")
        
        return n_dims
    
    def _compute_expansion_directions(self, n_dims: int) -> Optional[np.ndarray]:
        """
        Computa direções ótimas para expansão.
        
        Idealmente, encontra direções onde há muita "tensão"
        no espaço atual - gradientes fortes que não podem
        ser resolvidos nas dimensões existentes.
        
        TODO:
            - Implementar análise de gradientes
            - PCA dos resíduos
        """
        # Por agora, direções aleatórias ortogonais
        return None  # Manifold gerará aleatórias
    
    # =========================================================================
    # FASE 2: CONFIGURAÇÃO
    # =========================================================================
    
    def _configuration_phase(self, state: CycleState):
        """
        Fase de configuração: elementos se arranjam.
        
        Usa annealing para encontrar configuração de baixa energia.
        Os pontos "fluem" seguindo -∇F, e a temperatura diminui
        gradualmente.
        
        Esta é a fase "ilógico-lógica" - configurações emergem
        que parecem erradas mas resolvem algo.
        """
        logger.debug("Entrando fase de CONFIGURAÇÃO")
        
        T = self.config.configuration_temperature
        cooling_rate = self.config.configuration_cooling_rate
        
        for step in range(self.config.configuration_steps):
            # Ajusta temperatura
            self.field.set_temperature(T)
            
            # Para cada ponto ativo, dá um passo de descida
            active = self.manifold.get_active_points()
            
            for point_id, point in active:
                # Descida no gradiente de F
                new_coords = self.field.descend(
                    point.coordinates, 
                    step_size=0.01 * T,  # Step proporcional a T
                    steps=1
                )
                point.coordinates = new_coords
                
            # Atualiza deformações da métrica
            self.metric.relax(rate=0.1 * (1 - cooling_rate))
            
            # Reforça deformações nos pontos ativos
            for _, point in active:
                self.metric.deform_at_point(point)
                
            # Esfria
            T *= cooling_rate
            
            # Decai ativações levemente
            self.manifold.decay_activations(rate=0.01)
            
        # Retorna temperatura ao normal
        self.field.set_temperature(1.0)
        
        logger.debug(f"Configuração completa após {self.config.configuration_steps} passos")
    
    # =========================================================================
    # FASE 3: COMPRESSÃO
    # =========================================================================
    
    def _compression_phase(self, state: CycleState) -> int:
        """
        Fase de compressão: colapsa em estrutura densa.
        
        Compressão acontece quando energia livre está baixa,
        indicando que o sistema encontrou boa configuração
        e pode descartar dimensões extras.
        
        A compressão não é só remoção - ela redefine a própria
        métrica do espaço restante.
        
        Returns:
            Número de dimensões removidas
        """
        logger.debug("Entrando fase de COMPRESSÃO")
        
        # Decide se precisa comprimir
        current_F = self.field.stats()['mean_F']
        
        if current_F > self.config.compression_threshold:
            logger.debug(f"F={current_F:.4f} > threshold, skip compression")
            return 0
            
        # Quantas dimensões remover
        current_extra = self.manifold.current_dim - self.manifold.config.base_dim
        max_to_remove = current_extra - self.config.min_compression_dims
        
        if max_to_remove <= 0:
            logger.debug("Já no mínimo de dimensões")
            return 0
            
        n_dims = max(1, int(max_to_remove * self.config.compression_rate))
        n_dims = min(n_dims, max_to_remove)
        
        # Identifica quais dimensões são menos importantes
        # TODO: PCA ou análise de variância
        dims_to_remove = self._identify_compressible_dimensions(n_dims)
        
        # Antes de comprimir, cristaliza conhecimento no grafo
        self._crystallize_to_graph()
        
        # Comprime
        self.manifold.contract_dimension(n_dims)
        
        logger.debug(f"Comprimido {n_dims} dimensões. "
                    f"Nova dim: {self.manifold.current_dim}")
        
        return n_dims
    
    def _identify_compressible_dimensions(self, n_dims: int) -> List[int]:
        """
        Identifica dimensões que podem ser removidas.
        
        Critério: dimensões com menor variância nos pontos
        ativos contribuem menos para discriminação.
        
        TODO: Implementar análise real
        """
        # Por agora, remove as últimas (LIFO)
        current_dim = self.manifold.current_dim
        return list(range(current_dim - n_dims, current_dim))
    
    def _crystallize_to_graph(self):
        """
        Cristaliza estado do campo em estrutura de grafo.
        
        Antes de comprimir, salvamos o conhecimento aprendido
        em forma discreta (conexões no grafo Hebbiano).
        
        TODO:
            - Conectar com MycelialReasoning
            - Transformar atratores em nós
            - Transformar geodésicas em arestas
        """
        field_state = self.field.get_state()
        
        if field_state is None:
            return
            
        # Atratores viram nós importantes
        attractors = field_state.attractors
        
        # Geodésicas entre atratores viram arestas
        # TODO: Computar geodésicas entre pares de atratores
        
        logger.debug(f"Cristalizado {len(attractors)} atratores")
    
    # =========================================================================
    # FASE 4: META-ATUALIZAÇÃO
    # =========================================================================
    
    def _meta_update(self, state: CycleState) -> Dict:
        """
        Atualiza as regras de transição baseado no resultado do ciclo.
        
        Esta é a parte que faz o sistema realmente aprender:
        as regras de como expandir/comprimir evoluem baseado
        em quão bem os ciclos anteriores funcionaram.
        
        Returns:
            Dicionário com deltas aplicados às regras
        """
        logger.debug("Entrando META-ATUALIZAÇÃO")
        
        delta = {}
        
        # Avalia sucesso do ciclo
        # Sucesso = reduziu F sem perder informação importante
        F_delta = state.free_energy_delta
        
        # Se F diminuiu muito, esse ciclo foi bom
        # Reforça as decisões tomadas
        learning_signal = -F_delta  # Negativo porque queremos minimizar F
        
        # Atualiza pesos de expansão
        if state.dimensions_added > 0:
            expansion_success = learning_signal > 0
            adjustment = self.config.meta_learning_rate * (1 if expansion_success else -1)
            self.rule.expansion_weights *= (1 + adjustment)
            delta['expansion_adjustment'] = adjustment
            
        # Atualiza pesos de compressão
        if state.dimensions_removed > 0:
            compression_success = learning_signal > 0
            adjustment = self.config.meta_learning_rate * (1 if compression_success else -1)
            self.rule.compression_weights *= (1 + adjustment)
            delta['compression_adjustment'] = adjustment
            
        # Armazena no histórico da regra
        self.rule.history.append({
            'cycle': self.cycle_count,
            'F_delta': F_delta,
            'dims_added': state.dimensions_added,
            'dims_removed': state.dimensions_removed,
            'attractors': state.attractors_formed
        })
        
        # Mantém histórico limitado
        if len(self.rule.history) > self.config.history_length:
            self.rule.history = self.rule.history[-self.config.history_length:]
            
        logger.debug(f"Meta-update: signal={learning_signal:.4f}, delta={delta}")
        
        return delta
    
    # =========================================================================
    # CICLOS ESPECIAIS
    # =========================================================================
    
    def trigger_cycle(self, embedding: np.ndarray) -> CycleState:
        """
        Inicia um ciclo a partir de um trigger (embedding).
        
        Convenience method para run_cycle com trigger.
        """
        return self.run_cycle(trigger=embedding)
    
    def continuous_cycles(self, 
                         n_cycles: int,
                         triggers: Optional[List[np.ndarray]] = None) -> List[CycleState]:
        """
        Executa múltiplos ciclos em sequência.
        
        Args:
            n_cycles: Número de ciclos
            triggers: Lista de triggers (um por ciclo, opcional)
            
        Returns:
            Lista de CycleStates
        """
        states = []
        
        for i in range(n_cycles):
            trigger = triggers[i] if triggers and i < len(triggers) else None
            state = self.run_cycle(trigger=trigger)
            states.append(state)
            
        return states
    
    def until_stable(self, 
                     max_cycles: int = 100,
                     stability_threshold: float = 0.01) -> List[CycleState]:
        """
        Executa ciclos até o sistema estabilizar.
        
        Estabilidade = variação de F menor que threshold
        por N ciclos consecutivos.
        
        Args:
            max_cycles: Máximo de ciclos
            stability_threshold: Variação máxima de F para considerar estável
            
        Returns:
            Lista de CycleStates
        """
        states = []
        stable_count = 0
        
        for _ in range(max_cycles):
            state = self.run_cycle()
            states.append(state)
            
            if abs(state.free_energy_delta) < stability_threshold:
                stable_count += 1
                if stable_count >= 3:
                    logger.info(f"Sistema estabilizou após {len(states)} ciclos")
                    break
            else:
                stable_count = 0
                
        return states
    
    # =========================================================================
    # ESTADO E MÉTRICAS
    # =========================================================================
    
    def get_current_state(self) -> Dict:
        """Retorna estado atual do sistema."""
        return {
            'cycle_count': self.cycle_count,
            'current_phase': self.current_phase.value,
            'manifold_dim': self.manifold.current_dim,
            'base_dim': self.manifold.config.base_dim,
            'expansion_dims': self.manifold.current_dim - self.manifold.config.base_dim,
            'num_points': len(self.manifold.points),
            'field_stats': self.field.stats(),
            'metric_stats': self.metric.stats()
        }
    
    def get_history_summary(self) -> Dict:
        """Resumo do histórico de ciclos."""
        if not self.history:
            return {'cycles': 0}
            
        F_deltas = [s.free_energy_delta for s in self.history]
        dims_added = sum(s.dimensions_added for s in self.history)
        dims_removed = sum(s.dimensions_removed for s in self.history)
        
        return {
            'cycles': len(self.history),
            'total_F_delta': sum(F_deltas),
            'mean_F_delta': np.mean(F_deltas),
            'total_dims_added': dims_added,
            'total_dims_removed': dims_removed,
            'net_dim_change': dims_added - dims_removed,
            'final_dim': self.manifold.current_dim
        }
    
    def reset(self):
        """Reseta estado do ciclo (mantém manifold)."""
        self.current_phase = CyclePhase.IDLE
        self.cycle_count = 0
        self.history = []
        self.rule = TransitionRule()
