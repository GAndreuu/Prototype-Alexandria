"""
Nemesis Integration - Self-Feeding Loop
========================================

Integra os módulos teóricos Nemesis ao loop cognitivo:
- ActiveInferenceAgent: seleção de ações
- VariationalFreeEnergy: cálculo de F
- PredictiveCodingNetwork: modelo preditivo
- MetaHebbianPlasticity: plasticidade adaptativa

Autor: Alexandria Project
Data: 2024-12
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NemesisConfig:
    """Configuração do sistema Nemesis"""
    state_dim: int = 64
    observation_dim: int = 384
    use_active_inference: bool = True
    use_predictive_coding: bool = True
    use_free_energy: bool = True
    use_meta_hebbian: bool = True
    planning_horizon: int = 3
    num_action_samples: int = 10
    save_path: str = "data/nemesis_state.pkl"


@dataclass
class NemesisAction:
    """Ação selecionada pelo Nemesis"""
    action_type: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_free_energy: float = 0.0
    information_gain: float = 0.0
    risk: float = 0.0
    confidence: float = 0.5


@dataclass  
class NemesisState:
    """Estado interno do Nemesis"""
    beliefs: Dict[str, np.ndarray] = field(default_factory=dict)
    free_energy: float = 0.0
    prediction_error: float = 0.0
    complexity: float = 0.0
    accuracy: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class NemesisIntegration:
    """
    Orquestrador dos módulos Nemesis para o Self-Feeding Loop.
    
    Fluxo:
    1. Recebe gap/hipótese do AbductionEngine
    2. ActiveInference seleciona ação ótima
    3. Após execução, PredictiveCoding atualiza modelo
    4. FreeEnergy calcula "surpresa"
    5. MetaHebbian ajusta regras de plasticidade
    """
    
    def __init__(self, config: Optional[NemesisConfig] = None, topology_engine=None):
        self.config = config or NemesisConfig()
        self.topology_engine = topology_engine
        
        # Estado interno
        self.state = NemesisState()
        self.history: List[Dict] = []
        
        # Inicializar módulos
        self._init_modules()
        
        logger.info("🧠 NemesisIntegration inicializado")
    
    def _init_modules(self):
        """Inicializa os módulos Nemesis"""
        # Active Inference
        self.active_inference = None
        if self.config.use_active_inference:
            try:
                from core.learning.active_inference import (
                    ActiveInferenceAgent, 
                    ActiveInferenceConfig
                )
                ai_config = ActiveInferenceConfig(
                    state_dim=self.config.state_dim,
                    planning_horizon=self.config.planning_horizon,
                    num_action_samples=self.config.num_action_samples
                )
                self.active_inference = ActiveInferenceAgent(ai_config)
                logger.info("  ✅ ActiveInferenceAgent carregado")
            except Exception as e:
                logger.warning(f"  ⚠️ ActiveInference não disponível: {e}")
        
        # Free Energy
        self.free_energy = None
        if self.config.use_free_energy:
            try:
                from core.learning.free_energy import (
                    VariationalFreeEnergy,
                    FreeEnergyConfig
                )
                fe_config = FreeEnergyConfig(
                    state_dim=self.config.state_dim,
                    observation_dim=self.config.observation_dim
                )
                self.free_energy = VariationalFreeEnergy(fe_config)
                logger.info("  ✅ VariationalFreeEnergy carregado")
            except Exception as e:
                logger.warning(f"  ⚠️ FreeEnergy não disponível: {e}")
        
        # Predictive Coding
        self.predictive_coding = None
        if self.config.use_predictive_coding:
            try:
                from core.learning.predictive_coding import (
                    PredictiveCodingNetwork,
                    PredictiveCodingConfig
                )
                pc_config = PredictiveCodingConfig(
                    input_dim=self.config.observation_dim,
                    code_dim=self.config.state_dim
                )
                self.predictive_coding = PredictiveCodingNetwork(pc_config)
                logger.info("  ✅ PredictiveCodingNetwork carregado")
            except Exception as e:
                logger.warning(f"  ⚠️ PredictiveCoding não disponível: {e}")
        
        # Meta-Hebbian
        self.meta_hebbian = None
        if self.config.use_meta_hebbian:
            try:
                from core.learning.meta_hebbian import (
                    MetaHebbianPlasticity,
                    MetaHebbianConfig
                )
                mh_config = MetaHebbianConfig()
                self.meta_hebbian = MetaHebbianPlasticity(mh_config)
                logger.info("  ✅ MetaHebbianPlasticity carregado")
            except Exception as e:
                logger.warning(f"  ⚠️ MetaHebbian não disponível: {e}")
    
    def select_action(
        self, 
        gap: Dict, 
        hypotheses: List[Dict],
        context: Optional[Dict] = None
    ) -> NemesisAction:
        """
        Seleciona a melhor ação usando Active Inference.
        
        Args:
            gap: Gap detectado pelo AbductionEngine
            hypotheses: Lista de hipóteses candidatas
            context: Contexto adicional
            
        Returns:
            NemesisAction com a ação selecionada
        """
        if not hypotheses:
            return self._fallback_action(gap)
        
        # Se Active Inference disponível, usar para seleção
        if self.active_inference:
            try:
                return self._select_with_active_inference(gap, hypotheses, context)
            except Exception as e:
                logger.warning(f"Active Inference falhou: {e}")
        
        # Fallback: selecionar hipótese com maior confiança
        return self._select_by_confidence(hypotheses)
    
    def _select_with_active_inference(
        self, 
        gap: Dict, 
        hypotheses: List[Dict],
        context: Optional[Dict]
    ) -> NemesisAction:
        """Seleção via Active Inference"""
        best_action = None
        best_efe = float('inf')  # Queremos minimizar Expected Free Energy
        
        for hyp in hypotheses:
            # Converter hipótese para embedding se possível
            if self.topology_engine:
                try:
                    embedding = self.topology_engine.encode([hyp.get('hypothesis_text', '')])[0]
                except:
                    embedding = np.random.randn(self.config.observation_dim)
            else:
                embedding = np.random.randn(self.config.observation_dim)
            
            # Calcular Expected Free Energy para esta ação
            # G = Risk + Ambiguity
            # Risk: divergência de preferências
            # Ambiguity: incerteza sobre resultados
            
            confidence = hyp.get('confidence_score', 0.5)
            
            # Simplificação: EFE = -log(confidence) + incerteza
            risk = -np.log(confidence + 0.01)
            ambiguity = 1.0 - confidence
            efe = risk + ambiguity
            
            if efe < best_efe:
                best_efe = efe
                best_action = NemesisAction(
                    action_type=self._determine_action_type(confidence),
                    target=hyp.get('target_cluster', ''),
                    parameters={
                        'hypothesis_id': hyp.get('id', ''),
                        'source': hyp.get('source_cluster', ''),
                        'text': hyp.get('hypothesis_text', '')
                    },
                    expected_free_energy=efe,
                    information_gain=confidence * 0.5,
                    risk=risk,
                    confidence=confidence
                )
        
        return best_action or self._fallback_action(gap)
    
    def _determine_action_type(self, confidence: float) -> str:
        """Determina tipo de ação baseado em confiança"""
        if confidence >= 0.8:
            return "BRIDGE_CONCEPTS"
        elif confidence >= 0.6:
            return "DEEPEN_TOPIC"
        elif confidence >= 0.4:
            return "EXPLORE_CLUSTER"
        else:
            return "QUERY_SEARCH"
    
    def _select_by_confidence(self, hypotheses: List[Dict]) -> NemesisAction:
        """Seleção por maior confiança (fallback)"""
        best_hyp = max(hypotheses, key=lambda h: h.get('confidence_score', 0))
        confidence = best_hyp.get('confidence_score', 0.5)
        
        return NemesisAction(
            action_type=self._determine_action_type(confidence),
            target=best_hyp.get('target_cluster', ''),
            parameters={
                'hypothesis_id': best_hyp.get('id', ''),
                'source': best_hyp.get('source_cluster', ''),
                'text': best_hyp.get('hypothesis_text', '')
            },
            expected_free_energy=1.0 - confidence,
            confidence=confidence
        )
    
    def _fallback_action(self, gap: Dict) -> NemesisAction:
        """Ação de fallback quando não há hipóteses"""
        return NemesisAction(
            action_type="QUERY_SEARCH",
            target=gap.get('description', 'exploration'),
            parameters={'gap_id': gap.get('gap_id', '')},
            expected_free_energy=1.0,
            confidence=0.3
        )
    
    def update_after_action(
        self, 
        action: NemesisAction, 
        observation: np.ndarray,
        reward: float
    ) -> Dict[str, float]:
        """
        Atualiza modelos após execução de ação.
        
        Args:
            action: Ação executada
            observation: Embedding da evidência encontrada
            reward: Reward recebido (0-1)
            
        Returns:
            Métricas de atualização
        """
        metrics = {
            'free_energy': 0.0,
            'prediction_error': 0.0,
            'complexity': 0.0,
            'accuracy': reward
        }
        
        # 1. Atualizar Predictive Coding
        if self.predictive_coding:
            try:
                result = self.predictive_coding.process(observation)
                metrics['prediction_error'] = result.get('total_error', 0.0)
            except Exception as e:
                logger.debug(f"PredictiveCoding update falhou: {e}")
        
        # 2. Calcular Free Energy
        if self.free_energy:
            try:
                F, components = self.free_energy.compute(observation)
                metrics['free_energy'] = F
                metrics['complexity'] = components.get('complexity', 0.0)
                
                # Atualizar beliefs
                self.free_energy.update_beliefs(observation)
            except Exception as e:
                logger.debug(f"FreeEnergy compute falhou: {e}")
        
        # 3. Evoluir regras Meta-Hebbian baseado em fitness
        if self.meta_hebbian:
            try:
                # Fitness = -F (queremos minimizar F, então maximizar -F)
                fitness = -metrics['free_energy'] if metrics['free_energy'] > 0 else reward
                self.meta_hebbian.evolve_rules([fitness])
            except Exception as e:
                logger.debug(f"MetaHebbian evolve falhou: {e}")
        
        # Atualizar estado interno
        self.state.free_energy = metrics['free_energy']
        self.state.prediction_error = metrics['prediction_error']
        self.state.complexity = metrics['complexity']
        self.state.accuracy = reward
        self.state.timestamp = datetime.now()
        
        # Histórico
        self.history.append({
            'timestamp': self.state.timestamp.isoformat(),
            **metrics
        })
        
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais do Nemesis"""
        return {
            'free_energy': self.state.free_energy,
            'prediction_error': self.state.prediction_error,
            'complexity': self.state.complexity,
            'accuracy': self.state.accuracy,
            'modules_active': {
                'active_inference': self.active_inference is not None,
                'free_energy': self.free_energy is not None,
                'predictive_coding': self.predictive_coding is not None,
                'meta_hebbian': self.meta_hebbian is not None
            },
            'history_length': len(self.history)
        }
    
    def get_free_energy_trend(self, window: int = 10) -> float:
        """Calcula tendência de Free Energy (negativo = melhorando)"""
        if len(self.history) < 2:
            return 0.0
        
        recent = self.history[-window:]
        fe_values = [h.get('free_energy', 0) for h in recent]
        
        if len(fe_values) < 2:
            return 0.0
        
        # Regressão linear simples
        n = len(fe_values)
        x = np.arange(n)
        slope = (n * np.sum(x * fe_values) - np.sum(x) * np.sum(fe_values)) / \
                (n * np.sum(x**2) - np.sum(x)**2 + 1e-8)
        
        return slope


def create_nemesis_integration(
    topology_engine=None,
    use_all: bool = True
) -> NemesisIntegration:
    """Factory para criar NemesisIntegration"""
    config = NemesisConfig(
        use_active_inference=use_all,
        use_predictive_coding=use_all,
        use_free_energy=use_all,
        use_meta_hebbian=use_all
    )
    return NemesisIntegration(config, topology_engine)
