"""
Script de Execução Avançada: Self-Feeding Loop com Geometria Cognitiva
======================================================================

Este script substitui o run_real_loop.py para utilizar as novas integrações:
- NemesisBridgeIntegration (Planning Geométrico)
- LearningFieldIntegration (Predictive Coding no Manifold)
- GeodesicFlow (Caminhos Semânticos)

Funcionalidades:
1. Cálculo real de Prediction Error (Distance Geodésica)
2. Free Energy baseada na curvatura do manifold
3. Gaps reais do Abduction Engine
"""

import sys
import os
import argparse
import logging
import numpy as np
from datetime import datetime

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Imports Core
from core.loop.hypothesis_executor import HypothesisExecutor
from core.loop.feedback_collector import ActionFeedbackCollector
from core.loop.incremental_learner import IncrementalLearner
from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig
from core.loop.loop_metrics import LoopMetrics

# Imports Integrations
from core.integrations.nemesis_bridge_integration import NemesisBridgeIntegration, NemesisBridgeConfig
from core.integrations.learning_field_integration import LearningFieldIntegration, LearningFieldConfig
from core.integrations.geodesic_bridge_integration import GeodesicBridgeIntegration

# Imports Engines
from core.topology.topology_engine import TopologyEngine
from core.memory.semantic_memory import SemanticFileSystem
from core.reasoning.abduction_engine import AbductionEngine
from core.reasoning.neural_learner import V2Learner
from core.field.vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedAdapter:
    """Adapta as novas integrações para a interface esperada pelo Loop"""
    
    def __init__(self, nemesis_bridge: NemesisBridgeIntegration):
        self.nemesis = nemesis_bridge
        
    def select_action(self, loop_state):
        """Converte LoopState -> GeometricAction -> AgentAction"""
        # Extrair gaps e hypotheses do loop state
        gaps = loop_state.gaps
        hypotheses = loop_state.hypotheses
        
        # Selecionar ação geométrica
        geo_action = self.nemesis.select_action_geometric(gaps, hypotheses)
        
        # Converter para formato legado AgentAction se necessário
        # Mas o SelfFeedingLoop usa o retorno disto para criar hipótese
        # Vamos retornar um objeto compatível com AgentAction
        return type('AgentAction', (), {
            'action_type': type('ActionType', (), {'name': geo_action.action_type})(),
            'target': geo_action.target,
            'parameters': geo_action.parameters,
            'confidence': 0.8, # Estimado
            'expected_free_energy': geo_action.geometric_efe.total if geo_action.geometric_efe else 0.0
        })()

class AdvancedLoopIntegration:
    def __init__(self):
        self.topology = None
        self.memory = None
        self.abduction = None
        self.learner = None
        self.bridge = None
        self.nemesis_bridge = None
        self.learning_field = None
        self.loop = None
        
    def initialize(self, cycles: int = 100):
        logger.info("🚀 Inicializando Arquitetura Avançada...")
        
        # 1. Base Engines
        self.topology = TopologyEngine()
        self.memory = SemanticFileSystem(self.topology)
        self.abduction = AbductionEngine() # Assumindo que funciona sem args ou com args default
        self.learner = V2Learner(device="cpu") # Force CPU for safety
        
        # 2. Bridge & Manifold
        logger.info("🌌 Inicializando VQ-VAE Manifold Bridge...")
        self.bridge = VQVAEManifoldBridge(BridgeConfig())
        # Mock connect for demo if needed, or real load
        # self.bridge.connect_vqvae(...) 
        
        # 3. Integrations
        logger.info("🧠 Inicializando Integrações Cognitivas...")
        
        # Nemesis Bridge (Planning)
        self.nemesis_bridge = NemesisBridgeIntegration(
            bridge=self.bridge,
            config=NemesisBridgeConfig(
                use_geodesic_distance=True,
                curvature_weight=0.5
            )
        )
        
        # Learning Field (Predictive Coding)
        self.learning_field = LearningFieldIntegration(
            bridge=self.bridge,
            config=LearningFieldConfig(
                pc_use_geodesic_error=True
            )
        )
        
        # 4. Setup Loop
        self._setup_loop(cycles)
        return True
        
    def _setup_loop(self, cycles):
        # Executor e Collector padrão
        executor = HypothesisExecutor(self.memory, self.topology)
        collector = ActionFeedbackCollector(self.topology)
        learner_wrapper = IncrementalLearner(self.learner)
        
        # Adapter para Active Inference
        adapter = UnifiedAdapter(self.nemesis_bridge)
        
        # Config Loop
        config = LoopConfig(
            max_cycles=cycles,
            use_active_inference=True, # Usar Adapter
            stop_on_convergence=True,
            convergence_threshold=0.005 # Mais estrito
        )
        
        # Instanciar Loop
        self.loop = SelfFeedingLoop(
            abduction_engine=self.abduction,
            hypothesis_executor=executor,
            feedback_collector=collector,
            incremental_learner=learner_wrapper,
            active_inference_adapter=adapter,
            config=config,
            on_action_complete=self._on_action_complete,
            field=self.bridge # Passar bridge como 'field' para stats
        )
        
    def _on_action_complete(self, hypothesis, result, feedback):
        """Callback: Atualiza crenças geométricas após ação"""
        if not self.nemesis_bridge:
            return

        # Recuperar observação
        obs = np.zeros(384)
        if result.evidence_found:
             obs = self.topology.encode([str(result.evidence_found[0])])[0]
             
        # Reconstruir ação geométrica (simplificado)
        # O ideal seria ter a ação original, mas vamos reconstruir o contexto
        from core.integrations.nemesis_bridge_integration import GeometricAction, GeometricEFE
        
        # Criar objeto dummy EFE para evitar erro
        dummy_efe = GeometricEFE(0,0,0,0,0,0,0)
        
        action = GeometricAction(
            action_type=str(hypothesis.get('_action_type', 'UNKNOWN')),
            target=str(hypothesis.get('target_cluster', 'unknown')),
            parameters=hypothesis,
            geometric_efe=dummy_efe
        )
        
        # Update Geometric Beliefs (Calcula Prediction Error Real!)
        metrics = self.nemesis_bridge.update_beliefs_geometric(
            observation=obs,
            action=action,
            reward=feedback.reward_signal
        )
        
        # Update Learning Field (Calcula Free Energy Real!)
        lf_res = self.learning_field.process_observation(obs)
        
        logger.info(f"📐 Update Geométrico: PredErr={metrics.get('prediction_error', 0.0):.4f} | FreeEnergy={lf_res.get('free_energy', 0.0):.4f}")

    def run(self):
        return self.loop.run_continuous()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=100)
    args = parser.parse_args()
    
    system = AdvancedLoopIntegration()
    if system.initialize(args.cycles):
        system.run()
