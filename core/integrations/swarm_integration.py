"""
Swarm Integration Layer
=======================

Camada base que conecta o sistema de navegação Swarm aos módulos do Core.
Este é o ponto de entrada principal para integração Swarm ↔ Core.

Uso:
    from core.integrations import SwarmIntegration
    
    integration = SwarmIntegration(topology_engine=topology)
    result = integration.navigate_with_inference("quantum", "consciousness")
"""

import logging
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Imports condicionais para graceful degradation
try:
    from swarm import SwarmNavigator
    from swarm.core import NavigationMode, NavigationResult, Context
    SWARM_AVAILABLE = True
except ImportError:
    logger.warning("Swarm module not available. SwarmIntegration will use mock mode.")
    SWARM_AVAILABLE = False
    SwarmNavigator = None
    NavigationMode = None
    NavigationResult = None

try:
    from core.learning.active_inference import ActiveInferenceAgent
    ACTIVE_INFERENCE_AVAILABLE = True
except ImportError:
    logger.debug("ActiveInferenceAgent not available for mode selection.")
    ACTIVE_INFERENCE_AVAILABLE = False
    ActiveInferenceAgent = None

try:
    from core.field.free_energy_field import FreeEnergyField
    FIELD_AVAILABLE = True
except ImportError:
    logger.debug("FreeEnergyField not available.")
    FIELD_AVAILABLE = False
    FreeEnergyField = None


# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class SwarmModeRecommendation:
    """Recomendação de modo de navegação."""
    mode: str  # NavigationMode value
    confidence: float
    reasoning: str
    efe_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class IntegratedNavigationResult:
    """Resultado de navegação integrada Swarm + Core."""
    success: bool
    steps: int
    path: List[np.ndarray]
    init_similarity: float
    final_similarity: float
    improvement: float
    mode_used: str
    mode_recommendation: Optional[SwarmModeRecommendation] = None
    core_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmIntegrationConfig:
    """Configuração da integração Swarm ↔ Core."""
    memory_path: str = "data/swarm_memory.json"
    use_active_inference_mode: bool = True
    use_field_energy: bool = True
    default_mode: str = "balanced"
    max_steps: int = 50
    debug: bool = False


# =============================================================================
# INTEGRAÇÃO PRINCIPAL
# =============================================================================

class SwarmIntegration:
    """
    Camada de integração entre Swarm e Core.
    
    Conecta:
    - SwarmNavigator (navegação semântica)
    - ActiveInferenceAgent (seleção de modo)
    - FreeEnergyField (métricas geométricas)
    - TopologyEngine (embeddings compartilhados)
    
    Exemplo:
        integration = SwarmIntegration(
            topology_engine=topology,
            mycelial=mycelial_reasoning
        )
        result = integration.navigate_with_inference("A", "B")
    """
    
    def __init__(
        self,
        topology_engine=None,
        mycelial=None,
        active_inference: Optional[Any] = None,
        field: Optional[Any] = None,
        config: Optional[SwarmIntegrationConfig] = None
    ):
        """
        Inicializa a integração Swarm ↔ Core.
        
        Args:
            topology_engine: TopologyEngine para embeddings
            mycelial: MycelialReasoning para grafo Hebbian
            active_inference: ActiveInferenceAgent para seleção de modo
            field: FreeEnergyField para métricas geométricas
            config: Configuração da integração
        """
        self.config = config or SwarmIntegrationConfig()
        self.topology = topology_engine
        self.mycelial = mycelial
        self.active_inference = active_inference
        self.field = field
        
        # Auto-initialize Active Inference if not provided (fix Bug #2)
        if self.active_inference is None and ACTIVE_INFERENCE_AVAILABLE:
            try:
                self.active_inference = ActiveInferenceAgent()
                logger.info("Active Inference Agent auto-initialized")
            except Exception as e:
                logger.warning(f"Could not auto-initialize Active Inference: {e}")
                logger.warning("Falling back to heuristic mode selection")
        elif self.active_inference is None:
            logger.info("Active Inference not available, using heuristic mode selection")
        
        # Inicializar Swarm se disponível
        if SWARM_AVAILABLE:
            self.swarm = SwarmNavigator(
                topology_engine=topology_engine,
                mycelial_system=mycelial,
                memory_path=self.config.memory_path,
                use_neurodiverse=True
            )
            logger.info("SwarmIntegration initialized with real SwarmNavigator")
        else:
            self.swarm = None
            logger.warning("SwarmIntegration in mock mode (Swarm not available)")
        
        # Estatísticas
        self._stats = {
            "navigations": 0,
            "successful": 0,
            "modes_used": {},
            "avg_improvement": 0.0
        }
    
    def navigate(
        self,
        start_concept: Union[str, np.ndarray],
        target_concept: Union[str, np.ndarray],
        max_steps: Optional[int] = None,
        debug: Optional[bool] = None
    ) -> IntegratedNavigationResult:
        """
        Navegação básica sem seleção automática de modo.
        
        Args:
            start_concept: Conceito ou embedding inicial
            target_concept: Conceito ou embedding alvo
            max_steps: Máximo de passos (default: config.max_steps)
            debug: Modo debug (default: config.debug)
            
        Returns:
            IntegratedNavigationResult com métricas
        """
        if not self.swarm:
            return self._mock_navigation(start_concept, target_concept)
        
        max_steps = max_steps or self.config.max_steps
        debug = debug if debug is not None else self.config.debug
        
        result = self.swarm.navigate(
            start_concept=start_concept,
            target_concept=target_concept,
            max_steps=max_steps,
            debug=debug
        )
        
        return self._wrap_result(result, mode_used=self.config.default_mode)
    
    def navigate_with_inference(
        self,
        start_concept: Union[str, np.ndarray],
        target_concept: Union[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None
    ) -> IntegratedNavigationResult:
        """
        Navegação com seleção automática de modo via Active Inference.
        
        O modo é selecionado baseado em:
        - Expected Free Energy (risk + ambiguity)
        - Distância semântica entre conceitos
        - Contexto adicional (se fornecido)
        
        Args:
            start_concept: Conceito ou embedding inicial
            target_concept: Conceito ou embedding alvo
            context: Contexto adicional para seleção de modo
            max_steps: Máximo de passos
            
        Returns:
            IntegratedNavigationResult com métricas e recomendação de modo
        """
        # 1. Resolver embeddings
        start_emb = self._resolve_embedding(start_concept)
        target_emb = self._resolve_embedding(target_concept)
        
        # 2. Obter recomendação de modo
        mode_rec = self._get_mode_recommendation(start_emb, target_emb, context)
        
        # 3. Executar navegação
        if not self.swarm:
            result = self._mock_navigation(start_concept, target_concept)
            result.mode_recommendation = mode_rec
            return result
        
        max_steps = max_steps or self.config.max_steps
        
        # Ajustar max_steps baseado no modo
        if mode_rec.mode == "sprint":
            max_steps = min(max_steps, 20)
        elif mode_rec.mode == "cautious":
            max_steps = int(max_steps * 1.5)
        
        result = self.swarm.navigate(
            start_concept=start_concept,
            target_concept=target_concept,
            max_steps=max_steps,
            debug=self.config.debug
        )
        
        # 4. Wrap com métricas
        integrated = self._wrap_result(result, mode_used=mode_rec.mode)
        integrated.mode_recommendation = mode_rec
        
        # 5. Adicionar métricas do Core se disponíveis
        if self.field and FIELD_AVAILABLE:
            integrated.core_metrics["field_energy"] = self._compute_field_energy(
                integrated.path
            )
        
        # 6. Atualizar estatísticas
        self._update_stats(integrated)
        
        return integrated
    
    def _get_mode_recommendation(
        self,
        start_emb: np.ndarray,
        target_emb: np.ndarray,
        context: Optional[Dict] = None
    ) -> SwarmModeRecommendation:
        """Obtém recomendação de modo via Active Inference ou heurística."""
        
        # Se Active Inference disponível e configurado
        if (self.config.use_active_inference_mode and 
            self.active_inference and 
            ACTIVE_INFERENCE_AVAILABLE):
            return self._mode_from_active_inference(start_emb, target_emb, context)
        
        # Fallback: heurística baseada em distância
        return self._mode_from_heuristic(start_emb, target_emb)
    
    def _mode_from_active_inference(
        self,
        start_emb: np.ndarray,
        target_emb: np.ndarray,
        context: Optional[Dict] = None
    ) -> SwarmModeRecommendation:
        """Seleciona modo usando Active Inference (EFE)."""
        try:
            # EFE REAL: Cria Action object e chama compute_expected_free_energy
            from core.learning.active_inference import Action, ActionType as AIActionType
            
            # Calcular similaridade para estimar information gain
            dot = np.dot(start_emb, target_emb)
            norm = np.linalg.norm(start_emb) * np.linalg.norm(target_emb)
            similarity = float(dot / (norm + 1e-9))
            
            # Criar Action object para EFE
            action = Action(
                action_type=AIActionType.BRIDGE_CONCEPTS,
                target="navigation",
                parameters={'start': 'start_concept', 'target': 'target_concept'},
                expected_information_gain=1.0 - similarity  # Maior distância = mais info gain
            )
            
            # Atualizar estado do agente com embedding atual
            if hasattr(self.active_inference, 'current_state'):
                # Reduzir dimensão se necessário (embedding 384 → state_dim 64)
                if len(start_emb) > len(self.active_inference.current_state):
                    self.active_inference.current_state = start_emb[:len(self.active_inference.current_state)]
                else:
                    self.active_inference.current_state = start_emb
            
            # Chamar EFE real (embeddings 384 = state_dim 384)
            G, components = self.active_inference.compute_expected_free_energy(
                action=action,
                current_state=start_emb
            )
            
            risk = components['risk']
            ambiguity = components['ambiguity']
            
            logger.debug(f"EFE Real: G={G:.4f}, risk={risk:.4f}, ambiguity={ambiguity:.4f}")
            
            # Lógica de seleção
            total = risk + ambiguity
            ambiguity_ratio = ambiguity / (total + 1e-9)
            risk_ratio = risk / (total + 1e-9)
            
            if ambiguity_ratio > 0.6:
                mode = "creative"
                reasoning = "Alta incerteza → exploração ampla"
            elif risk_ratio > 0.7:
                mode = "cautious"
                reasoning = "Alto risco → navegação cuidadosa"
            elif total < 0.3:
                mode = "sprint"
                reasoning = "Baixa energia livre → caminho direto"
            else:
                mode = "balanced"
                reasoning = "Condições normais → modo padrão"
            
            return SwarmModeRecommendation(
                mode=mode,
                confidence=max(0.0, 1.0 - total),
                reasoning=reasoning,
                efe_metrics={"risk": risk, "ambiguity": ambiguity, "total": total}
            )
            
        except Exception as e:
            logger.warning(f"Active Inference mode selection failed: {e}")
            return self._mode_from_heuristic(start_emb, target_emb)
    
    def _mode_from_heuristic(
        self,
        start_emb: np.ndarray,
        target_emb: np.ndarray
    ) -> SwarmModeRecommendation:
        """Seleciona modo via heurística simples (distância)."""
        # Calcular similaridade
        dot = np.dot(start_emb, target_emb)
        norm = np.linalg.norm(start_emb) * np.linalg.norm(target_emb)
        similarity = float(dot / (norm + 1e-9))
        
        if similarity > 0.7:
            mode = "sprint"
            reasoning = "Conceitos próximos → caminho direto"
        elif similarity < 0.3:
            mode = "creative"
            reasoning = "Conceitos distantes → exploração necessária"
        else:
            mode = "balanced"
            reasoning = "Distância moderada → modo padrão"
        
        return SwarmModeRecommendation(
            mode=mode,
            confidence=abs(similarity - 0.5) * 2,  # Mais confiança nos extremos
            reasoning=reasoning,
            efe_metrics={"similarity": similarity}
        )
    
    def _resolve_embedding(self, concept: Union[str, np.ndarray]) -> np.ndarray:
        """Resolve conceito para embedding."""
        if isinstance(concept, np.ndarray):
            return concept
        
        if self.topology and hasattr(self.topology, 'encode'):
            embeddings = self.topology.encode([concept])
            return embeddings[0]
        
        # Fallback: hash determinístico
        np.random.seed(hash(concept) % (2**32))
        return np.random.randn(384).astype(np.float32)
    
    def _compute_field_energy(self, path: List[np.ndarray]) -> Dict[str, float]:
        """Computa energia do campo ao longo do caminho."""
        if not path or not self.field:
            return {}
        
        try:
            energies = []
            for point in path:
                if hasattr(self.field, 'compute_energy'):
                    energy = self.field.compute_energy(point)
                    energies.append(float(energy))
            
            if energies:
                return {
                    "mean_energy": np.mean(energies),
                    "max_energy": np.max(energies),
                    "energy_variance": np.var(energies)
                }
        except Exception as e:
            logger.debug(f"Field energy computation failed: {e}")
        
        return {}
    
    def _wrap_result(
        self,
        result: Any,
        mode_used: str
    ) -> IntegratedNavigationResult:
        """Converte resultado do Swarm para IntegratedNavigationResult."""
        if hasattr(result, 'success'):
            return IntegratedNavigationResult(
                success=result.success,
                steps=result.steps,
                path=result.path if hasattr(result, 'path') else [],
                init_similarity=getattr(result, 'init_similarity', 0.0),
                final_similarity=getattr(result, 'final_similarity', 0.0),
                improvement=getattr(result, 'improvement', 0.0),
                mode_used=mode_used
            )
        
        # Dict fallback
        return IntegratedNavigationResult(
            success=result.get('success', False),
            steps=result.get('steps', 0),
            path=result.get('path', []),
            init_similarity=result.get('init_similarity', 0.0),
            final_similarity=result.get('final_similarity', 0.0),
            improvement=result.get('improvement', 0.0),
            mode_used=mode_used
        )
    
    def _mock_navigation(
        self,
        start: Union[str, np.ndarray],
        target: Union[str, np.ndarray]
    ) -> IntegratedNavigationResult:
        """Navegação mock para quando Swarm não está disponível."""
        start_emb = self._resolve_embedding(start)
        target_emb = self._resolve_embedding(target)
        
        # Simular caminho
        path = [start_emb]
        current = start_emb.copy()
        for _ in range(5):
            current = current + 0.2 * (target_emb - current)
            path.append(current.copy())
        
        init_sim = float(np.dot(start_emb, target_emb) / 
                        (np.linalg.norm(start_emb) * np.linalg.norm(target_emb) + 1e-9))
        final_sim = float(np.dot(current, target_emb) /
                         (np.linalg.norm(current) * np.linalg.norm(target_emb) + 1e-9))
        
        return IntegratedNavigationResult(
            success=True,
            steps=5,
            path=path,
            init_similarity=init_sim,
            final_similarity=final_sim,
            improvement=final_sim - init_sim,
            mode_used="mock"
        )
    
    def _update_stats(self, result: IntegratedNavigationResult):
        """Atualiza estatísticas internas."""
        self._stats["navigations"] += 1
        if result.success:
            self._stats["successful"] += 1
        
        mode = result.mode_used
        self._stats["modes_used"][mode] = self._stats["modes_used"].get(mode, 0) + 1
        
        # Running average
        n = self._stats["navigations"]
        old_avg = self._stats["avg_improvement"]
        self._stats["avg_improvement"] = old_avg + (result.improvement - old_avg) / n
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da integração."""
        return {
            **self._stats,
            "swarm_available": self.swarm is not None,
            "active_inference_available": self.active_inference is not None,
            "field_available": self.field is not None,
            "success_rate": (self._stats["successful"] / max(1, self._stats["navigations"]))
        }
    
    def reset_stats(self):
        """Reseta estatísticas."""
        self._stats = {
            "navigations": 0,
            "successful": 0,
            "modes_used": {},
            "avg_improvement": 0.0
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_swarm_integration(
    topology_engine=None,
    mycelial=None,
    active_inference=None,
    config: Optional[SwarmIntegrationConfig] = None
) -> SwarmIntegration:
    """
    Factory function para criar SwarmIntegration.
    
    Args:
        topology_engine: TopologyEngine para embeddings
        mycelial: MycelialReasoning para grafo Hebbian
        active_inference: ActiveInferenceAgent para seleção de modo
        config: Configuração opcional
        
    Returns:
        SwarmIntegration configurado
    """
    return SwarmIntegration(
        topology_engine=topology_engine,
        mycelial=mycelial,
        active_inference=active_inference,
        config=config
    )
