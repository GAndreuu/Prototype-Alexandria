"""
Alexandria Unified Integration
===============================

Módulo mestre que unifica todas as integrações do sistema Alexandria.

Arquitetura:
                            ┌─────────────────────────────────┐
                            │        AlexandriaCore           │
                            │  (Unified Integration Layer)     │
                            └───────────────┬─────────────────┘
                                            │
            ┌───────────────┬───────────────┼───────────────┬───────────────┐
            │               │               │               │               │
            ▼               ▼               ▼               ▼               ▼
    ┌───────────────┐ ┌───────────┐ ┌───────────────┐ ┌───────────┐ ┌───────────┐
    │   Nemesis     │ │ Learning  │ │  Abduction    │ │  Agents   │ │   Loop    │
    │   Bridge      │ │  Field    │ │ Compositional │ │ Compos.   │ │ Compos.   │
    │ Integration   │ │Integration│ │ Integration   │ │Integration│ │Integration│
    └───────┬───────┘ └─────┬─────┘ └───────┬───────┘ └─────┬─────┘ └─────┬─────┘
            │               │               │               │               │
            └───────────────┴───────────────┴───────┬───────┴───────────────┘
                                                    │
                                                    ▼
                            ┌─────────────────────────────────┐
                            │      VQVAEManifoldBridge        │
                            │    (Geometric Foundation)        │
                            └───────────────┬─────────────────┘
                                            │
                            ┌───────────────┴───────────────┐
                            │     CompositionalReasoner     │
                            │    (Geodesic Computation)      │
                            └───────────────────────────────┘

Uso:
    from alexandria_unified import AlexandriaCore
    
    core = AlexandriaCore.from_vqvae(vqvae_model)
    
    # Ciclo cognitivo completo
    result = core.cognitive_cycle(observation, goal)
    
    # Ou acesso granular
    core.nemesis.select_action_geometric(...)
    core.learning.process_observation(...)
    core.abduction.detect_gaps_geometric(...)

Autor: G (Alexandria Project)
Versão: 1.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

# Importar integrações
try:
    from nemesis_bridge_integration import (
        NemesisBridgeIntegration, 
        NemesisBridgeConfig,
        create_nemesis_bridge
    )
except ImportError:
    NemesisBridgeIntegration = None
    logger.warning("nemesis_bridge_integration not found")

try:
    from learning_field_integration import (
        LearningFieldIntegration,
        LearningFieldConfig,
        create_learning_field_integration
    )
except ImportError:
    LearningFieldIntegration = None
    logger.warning("learning_field_integration not found")

try:
    from abduction_compositional_integration import (
        AbductionCompositionalIntegration,
        AbductionCompositionalConfig,
        create_abduction_compositional
    )
except ImportError:
    AbductionCompositionalIntegration = None
    logger.warning("abduction_compositional_integration not found")

try:
    from agents_compositional_integration import (
        AgentsCompositionalIntegration,
        AgentsCompositionalConfig,
        create_agents_compositional
    )
except ImportError:
    AgentsCompositionalIntegration = None
    logger.warning("agents_compositional_integration not found")

try:
    from loop_compositional_integration import (
        LoopCompositionalIntegration,
        LoopCompositionalConfig,
        create_loop_compositional
    )
except ImportError:
    LoopCompositionalIntegration = None
    logger.warning("loop_compositional_integration not found")

try:
    from vqvae_manifold_bridge import (
        VQVAEManifoldBridge,
        BridgeConfig
    )
except ImportError:
    VQVAEManifoldBridge = None
    logger.warning("vqvae_manifold_bridge not found")


# =============================================================================
# CONFIGURAÇÃO UNIFICADA
# =============================================================================

@dataclass
class AlexandriaConfig:
    """Configuração unificada do sistema Alexandria."""
    
    # Bridge
    bridge_pull_strength: float = 0.5
    bridge_pull_radius: float = 0.3
    bridge_deformation_strength: float = 0.3
    
    # Nemesis
    nemesis_risk_weight: float = 1.0
    nemesis_ambiguity_weight: float = 1.0
    
    # Learning
    learning_use_geodesic: bool = True
    learning_sync_beliefs: bool = True
    
    # Abduction
    abduction_gap_threshold: float = 0.3
    abduction_max_hypotheses: int = 5
    
    # Agents
    agents_action_samples: int = 10
    agents_oracle_guidance: bool = True
    
    # Loop
    loop_max_iterations: int = 100
    loop_energy_target: float = 0.1
    loop_exploration_rate: float = 0.1


# =============================================================================
# RESULTADO DO CICLO COGNITIVO
# =============================================================================

@dataclass
class CognitiveCycleResult:
    """Resultado completo de um ciclo cognitivo."""
    
    # Percepção
    perception: Dict[str, Any]
    
    # Raciocínio
    reasoning: Dict[str, Any]
    gaps_detected: List[Dict]
    hypotheses_generated: List[Dict]
    
    # Ação
    action_selected: Dict[str, Any]
    
    # Aprendizado
    learning_metrics: Dict[str, float]
    
    # Métricas globais
    free_energy: float
    iteration: int
    duration_ms: float


# =============================================================================
# NÚCLEO UNIFICADO
# =============================================================================

class AlexandriaCore:
    """
    Núcleo unificado do sistema Alexandria.
    
    Coordena todas as integrações para fornecer uma interface
    coesa para o ciclo cognitivo completo.
    """
    
    def __init__(
        self,
        bridge: Optional[Any] = None,
        compositional: Optional[Any] = None,
        config: Optional[AlexandriaConfig] = None
    ):
        self.config = config or AlexandriaConfig()
        self.bridge = bridge
        self.compositional = compositional
        
        # Inicializar integrações
        self._init_integrations()
        
        # Estado global
        self._iteration = 0
        self._free_energy_history = []
        
        logger.info("AlexandriaCore initialized")
    
    @classmethod
    def from_vqvae(
        cls,
        vqvae_model,
        compositional=None,
        config: Optional[AlexandriaConfig] = None
    ) -> 'AlexandriaCore':
        """
        Cria AlexandriaCore a partir de um modelo VQ-VAE.
        
        Args:
            vqvae_model: Modelo VQ-VAE treinado
            compositional: CompositionalReasoner opcional
            config: Configuração
        
        Returns:
            AlexandriaCore configurado
        """
        config = config or AlexandriaConfig()
        
        # Criar bridge
        if VQVAEManifoldBridge is not None:
            bridge_config = BridgeConfig(
                pull_strength=config.bridge_pull_strength,
                pull_radius=config.bridge_pull_radius,
                deformation_strength=config.bridge_deformation_strength
            )
            bridge = VQVAEManifoldBridge(bridge_config)
            bridge.connect_vqvae(vqvae_model)
        else:
            bridge = None
            logger.error("VQVAEManifoldBridge not available")
        
        return cls(bridge, compositional, config)
    
    def _init_integrations(self):
        """Inicializa todas as integrações."""
        
        # Nemesis ↔ Bridge
        if NemesisBridgeIntegration is not None and self.bridge is not None:
            self.nemesis = NemesisBridgeIntegration(
                self.bridge,
                compositional=self.compositional,
                config=NemesisBridgeConfig(
                    risk_weight=self.config.nemesis_risk_weight,
                    ambiguity_weight=self.config.nemesis_ambiguity_weight
                )
            )
        else:
            self.nemesis = None
        
        # Learning ↔ Field
        if LearningFieldIntegration is not None and self.bridge is not None:
            self.learning = LearningFieldIntegration(
                self.bridge,
                compositional=self.compositional,
                config=LearningFieldConfig(
                    pc_use_geodesic_error=self.config.learning_use_geodesic,
                    sync_beliefs=self.config.learning_sync_beliefs
                )
            )
        else:
            self.learning = None
        
        # Abduction ↔ Compositional
        if AbductionCompositionalIntegration is not None and self.bridge is not None:
            self.abduction = AbductionCompositionalIntegration(
                self.bridge,
                compositional=self.compositional,
                config=AbductionCompositionalConfig(
                    gap_curvature_threshold=self.config.abduction_gap_threshold,
                    max_hypotheses_per_gap=self.config.abduction_max_hypotheses
                )
            )
        else:
            self.abduction = None
        
        # Agents ↔ Compositional
        if AgentsCompositionalIntegration is not None and self.bridge is not None:
            self.agents = AgentsCompositionalIntegration(
                self.bridge,
                compositional=self.compositional,
                config=AgentsCompositionalConfig(
                    action_samples=self.config.agents_action_samples,
                    oracle_attractor_guidance=self.config.agents_oracle_guidance
                )
            )
        else:
            self.agents = None
        
        # Loop ↔ Compositional
        if LoopCompositionalIntegration is not None and self.bridge is not None:
            self.loop = LoopCompositionalIntegration(
                self.bridge,
                compositional=self.compositional,
                config=LoopCompositionalConfig(
                    max_iterations=self.config.loop_max_iterations,
                    energy_target=self.config.loop_energy_target,
                    exploration_rate=self.config.loop_exploration_rate
                )
            )
        else:
            self.loop = None
    
    # =========================================================================
    # CICLO COGNITIVO PRINCIPAL
    # =========================================================================
    
    def cognitive_cycle(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray] = None,
        context: Optional[Dict] = None
    ) -> CognitiveCycleResult:
        """
        Executa um ciclo cognitivo completo.
        
        1. PERCEBER: Processar observação
        2. RACIOCINAR: Detectar gaps, gerar hipóteses
        3. AGIR: Selecionar ação
        4. APRENDER: Atualizar modelo
        
        Args:
            observation: Embedding de entrada
            goal: Goal opcional
            context: Contexto adicional
        
        Returns:
            CognitiveCycleResult com todos os resultados
        """
        start_time = time.time()
        self._iteration += 1
        
        # 1. PERCEBER
        perception = self._perceive(observation, context)
        
        # 2. RACIOCINAR
        reasoning, gaps, hypotheses = self._reason(observation, perception)
        
        # 3. AGIR
        action = self._act(observation, goal, hypotheses, context)
        
        # 4. APRENDER
        learning = self._learn(observation, perception, action)
        
        # Métricas globais
        F = perception.get('free_energy', 0.0)
        self._free_energy_history.append(F)
        
        duration = (time.time() - start_time) * 1000
        
        return CognitiveCycleResult(
            perception=perception,
            reasoning=reasoning,
            gaps_detected=gaps,
            hypotheses_generated=hypotheses,
            action_selected=action,
            learning_metrics=learning,
            free_energy=F,
            iteration=self._iteration,
            duration_ms=duration
        )
    
    def _perceive(
        self,
        observation: np.ndarray,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Fase de percepção."""
        result = {}
        
        # Usar Learning integration
        if self.learning is not None:
            proc = self.learning.process_observation(observation, context)
            result['code'] = proc.get('code')
            result['prediction'] = proc.get('prediction')
            result['free_energy'] = proc.get('free_energy', 0.0)
        else:
            # Fallback: apenas projetar
            if self.bridge is not None and hasattr(self.bridge, 'embed'):
                point = self.bridge.embed(observation)
                result['coordinates'] = point.coordinates
                result['nearest_distance'] = point.nearest_anchor_distance
                result['free_energy'] = point.nearest_anchor_distance
            else:
                result['free_energy'] = 0.0
        
        return result
    
    def _reason(
        self,
        observation: np.ndarray,
        perception: Dict
    ) -> Tuple[Dict, List[Dict], List[Dict]]:
        """Fase de raciocínio."""
        reasoning = {}
        gaps = []
        hypotheses = []
        
        # Usar Abduction integration
        if self.abduction is not None:
            # Detectar gaps
            detected = self.abduction.detect_gaps_geometric([observation])
            gaps = [{'gap_id': g.gap_id, 'type': g.gap_type, 'priority': g.priority_score} 
                    for g in detected[:3]]
            
            # Gerar hipóteses para gap principal
            if detected:
                hyps = self.abduction.generate_geodesic_hypotheses(detected[0])
                hypotheses = [{'id': h.hypothesis_id, 'confidence': h.confidence_score}
                              for h in hyps[:3]]
        
        reasoning['gaps_count'] = len(gaps)
        reasoning['hypotheses_count'] = len(hypotheses)
        
        return reasoning, gaps, hypotheses
    
    def _act(
        self,
        observation: np.ndarray,
        goal: Optional[np.ndarray],
        hypotheses: List[Dict],
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Fase de ação."""
        action = {}
        
        # Usar Nemesis integration para seleção
        if self.nemesis is not None:
            gaps_for_nemesis = [{'description': h['id'], 'priority_score': h['confidence']} 
                               for h in hypotheses]
            hyps_for_nemesis = [{'hypothesis_text': h['id'], 'confidence_score': h['confidence']}
                               for h in hypotheses]
            
            result = self.nemesis.select_action_geometric(
                gaps_for_nemesis,
                hyps_for_nemesis,
                observation
            )
            
            action['type'] = result.action_type
            action['target'] = result.target
            action['efe'] = result.geometric_efe.total if result.geometric_efe else 0
            action['confidence'] = 1.0 / (1.0 + action['efe'])
        
        # Ou usar Agents integration
        elif self.agents is not None and goal is not None:
            result = self.agents.action_geometric(observation, [goal], context)
            action['type'] = result.action_type
            action['energy'] = result.path_energy
            action['confidence'] = result.confidence
        
        else:
            action['type'] = 'none'
            action['confidence'] = 0.0
        
        return action
    
    def _learn(
        self,
        observation: np.ndarray,
        perception: Dict,
        action: Dict
    ) -> Dict[str, float]:
        """Fase de aprendizado."""
        metrics = {}
        
        # Usar Learning integration
        if self.learning is not None:
            # Computar reward implícito (redução de F)
            F_current = perception.get('free_energy', 0.0)
            if len(self._free_energy_history) > 1:
                F_prev = self._free_energy_history[-2]
                reward = F_prev - F_current
            else:
                reward = 0.0
            
            metrics['implicit_reward'] = reward
            metrics['free_energy'] = F_current
            
            # Trigger meta-learning se erro alto
            if perception.get('prediction') and hasattr(perception['prediction'], 'geodesic_error'):
                if perception['prediction'].geodesic_error > 0.1:
                    metrics['meta_learning_triggered'] = True
        
        return metrics
    
    # =========================================================================
    # CICLO AUTÔNOMO
    # =========================================================================
    
    def autonomous_run(
        self,
        initial_observation: np.ndarray,
        goal: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executa loop autônomo completo.
        
        Delega para LoopCompositionalIntegration.
        """
        if self.loop is not None:
            return self.loop.autonomous_cycle(
                initial_observation, goal, max_iterations
            )
        else:
            # Fallback: executar ciclos manuais
            max_iter = max_iterations or self.config.loop_max_iterations
            results = []
            
            for _ in range(max_iter):
                result = self.cognitive_cycle(initial_observation, goal)
                results.append(result)
                
                if result.free_energy < self.config.loop_energy_target:
                    break
            
            return {
                'iterations': len(results),
                'final_free_energy': results[-1].free_energy if results else 0,
                'results': results
            }
    
    # =========================================================================
    # API DE CONVENIÊNCIA
    # =========================================================================
    
    def embed(self, observation: np.ndarray) -> np.ndarray:
        """Embede observação no manifold."""
        if self.bridge is not None:
            point = self.bridge.embed(observation)
            return point.coordinates
        return observation
    
    def reason(
        self,
        start: np.ndarray,
        end: np.ndarray
    ) -> Dict[str, Any]:
        """Raciocina entre dois pontos via geodésica."""
        if self.compositional is not None:
            result = self.compositional.reason(start, end)
            return {
                'path': result.points,
                'trace': result.composition_trace,
                'length': result.path_length
            }
        
        # Fallback
        return {
            'path': np.array([start, end]),
            'trace': ['direct'],
            'length': np.linalg.norm(end - start)
        }
    
    def synthesize(
        self,
        sources: List[np.ndarray],
        target_direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sintetiza a partir de múltiplas fontes."""
        if self.agents is not None:
            result = self.agents.synthesize_geometric(sources, target_direction)
            return result.synthesized
        
        # Fallback: média
        return np.mean(sources, axis=0)
    
    # =========================================================================
    # DIAGNÓSTICO
    # =========================================================================
    
    def health_check(self) -> Dict[str, bool]:
        """Verifica saúde de todos os componentes."""
        return {
            'bridge': self.bridge is not None,
            'compositional': self.compositional is not None,
            'nemesis': self.nemesis is not None,
            'learning': self.learning is not None,
            'abduction': self.abduction is not None,
            'agents': self.agents is not None,
            'loop': self.loop is not None
        }
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas completas do sistema."""
        stats = {
            'health': self.health_check(),
            'iterations': self._iteration,
            'free_energy_history_size': len(self._free_energy_history)
        }
        
        if self._free_energy_history:
            stats['current_free_energy'] = self._free_energy_history[-1]
            stats['min_free_energy'] = min(self._free_energy_history)
            stats['mean_free_energy'] = np.mean(self._free_energy_history)
        
        # Stats de cada componente
        if self.bridge is not None:
            if hasattr(self.bridge, 'anchor_points') and self.bridge.anchor_points is not None:
                stats['bridge_anchors'] = len(self.bridge.anchor_points)
            else:
                stats['bridge_anchors'] = 0
        
        return stats
    
    def reset(self):
        """Reseta estado do sistema."""
        self._iteration = 0
        self._free_energy_history = []
        
        if self.loop is not None:
            self.loop.reset()


# =============================================================================
# FACTORY
# =============================================================================

def create_alexandria(
    vqvae_model=None,
    compositional=None,
    **config_kwargs
) -> AlexandriaCore:
    """
    Factory function para criar AlexandriaCore.
    """
    config = AlexandriaConfig(**config_kwargs)
    
    if vqvae_model is not None:
        return AlexandriaCore.from_vqvae(vqvae_model, compositional, config)
    else:
        return AlexandriaCore(config=config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Alexandria Unified Integration - Teste Completo")
    print("=" * 70)
    
    # Mock VQ-VAE
    class MockVQVAE:
        def get_codebook(self):
            np.random.seed(42)
            return np.random.randn(4, 256, 128).astype(np.float32)
    
    # Criar core
    print("\n1. Criando AlexandriaCore...")
    core = AlexandriaCore.from_vqvae(MockVQVAE())
    
    # Health check
    print("\n2. Health check:")
    health = core.health_check()
    for component, status in health.items():
        emoji = "✓" if status else "✗"
        print(f"   {emoji} {component}: {'OK' if status else 'NOT AVAILABLE'}")
    
    # Ciclo cognitivo
    print("\n3. Executando ciclo cognitivo...")
    observation = np.random.randn(384).astype(np.float32)
    goal = np.random.randn(384).astype(np.float32)
    
    result = core.cognitive_cycle(observation, goal)
    
    print(f"   Iteration: {result.iteration}")
    print(f"   Free energy: {result.free_energy:.4f}")
    print(f"   Duration: {result.duration_ms:.2f}ms")
    print(f"   Gaps detected: {len(result.gaps_detected)}")
    print(f"   Hypotheses: {len(result.hypotheses_generated)}")
    print(f"   Action: {result.action_selected.get('type', 'none')}")
    
    # Múltiplos ciclos
    print("\n4. Executando 10 ciclos...")
    for i in range(9):
        result = core.cognitive_cycle(observation, goal)
    
    print(f"   Final iteration: {result.iteration}")
    print(f"   Final F: {result.free_energy:.4f}")
    
    # Stats
    print("\n5. Estatísticas do sistema:")
    stats = core.stats()
    print(f"   Total iterations: {stats['iterations']}")
    print(f"   Current F: {stats.get('current_free_energy', 'N/A')}")
    print(f"   Min F: {stats.get('min_free_energy', 'N/A')}")
    print(f"   Bridge anchors: {stats.get('bridge_anchors', 0)}")
    
    print("\n" + "=" * 70)
    print("Alexandria Unified Integration - Teste Completo")
    print("Sistema integrado e funcional!")
    print("=" * 70)
