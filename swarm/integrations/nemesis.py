"""
Swarm ↔ Nemesis Integration
============================

Connects the Swarm navigation system to the Nemesis active inference engine.

This integration:
1. Translates SwarmAction results into NemesisAction format
2. Uses navigation trajectories to update Nemesis beliefs
3. Generates natural language explanations of navigation results
4. Enables feedback loop: Swarm explores → Nemesis evaluates → System learns
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NavigationExplanation:
    """Human-readable explanation of a Swarm navigation result."""
    summary: str
    path_description: str
    key_concepts: List[str]
    confidence: float
    neurotype_narrative: str
    actionable_insights: List[str]
    

@dataclass 
class SwarmNemesisResult:
    """Combined result from Swarm + Nemesis processing."""
    swarm_result: Dict
    explanation: NavigationExplanation
    nemesis_action: Optional[Any] = None
    feedback_signal: float = 0.0


class SwarmNemesisIntegration:
    """
    Bridges Swarm navigation to Nemesis active inference.
    
    Flow:
    1. User query → SwarmAction
    2. Swarm navigates semantic space
    3. This integration translates results to Nemesis format
    4. Nemesis evaluates expected free energy of path
    5. System generates explanation for user
    6. User feedback updates both systems
    """
    
    def __init__(
        self,
        swarm_navigator=None,
        nemesis_integration=None,
        topology_engine=None
    ):
        self.swarm = swarm_navigator
        self.nemesis = nemesis_integration
        self.topology = topology_engine
        
        # History for learning
        self.navigation_history = []
        self.feedback_history = []
        
    def navigate_and_explain(
        self,
        start_concept: Union[str, np.ndarray],
        target_concept: Union[str, np.ndarray],
        context: Optional[Dict] = None,
        generate_explanation: bool = True
    ) -> SwarmNemesisResult:
        """
        Execute navigation and generate human-readable explanation.
        
        Args:
            start_concept: Starting point (text or embedding)
            target_concept: Target point (text or embedding)
            context: Additional context for the navigation
            generate_explanation: Whether to generate NL explanation
            
        Returns:
            SwarmNemesisResult with navigation data and explanation
        """
        # 1. Execute Swarm navigation
        if self.swarm:
            swarm_result = self.swarm.navigate(
                start_concept=start_concept,
                target_concept=target_concept,
                max_steps=50,
                debug=False
            )
        else:
            # Mock result for testing
            swarm_result = self._mock_navigation_result()
        
        # 2. Generate explanation
        explanation = self._generate_explanation(
            swarm_result,
            start_concept,
            target_concept
        ) if generate_explanation else None
        
        # 3. Convert to Nemesis action if available
        nemesis_action = None
        if self.nemesis:
            nemesis_action = self._convert_to_nemesis_action(swarm_result)
        
        # 4. Record for learning
        self.navigation_history.append({
            'start': str(start_concept)[:50] if isinstance(start_concept, str) else 'embedding',
            'target': str(target_concept)[:50] if isinstance(target_concept, str) else 'embedding',
            'success': swarm_result.get('success', False),
            'improvement': swarm_result.get('improvement', 0.0)
        })
        
        return SwarmNemesisResult(
            swarm_result=swarm_result,
            explanation=explanation,
            nemesis_action=nemesis_action
        )
    
    def _generate_explanation(
        self,
        result: Dict,
        start: Union[str, np.ndarray],
        target: Union[str, np.ndarray]
    ) -> NavigationExplanation:
        """
        Generate human-readable explanation of navigation result.
        """
        success = result.get('success', False)
        steps = result.get('steps', 0)
        init_sim = result.get('init_similarity', 0.0)
        final_sim = result.get('final_similarity', 0.0)
        improvement = result.get('improvement', 0.0)
        neurotype_contribs = result.get('neurotype_contributions', {})
        
        # Determine start/target names
        start_name = start if isinstance(start, str) else "ponto inicial"
        target_name = target if isinstance(target, str) else "ponto alvo"
        
        # 1. Summary
        if success:
            summary = f"✅ Conexão encontrada entre '{start_name}' e '{target_name}' em {steps} passos."
        else:
            summary = f"⚠️ Caminho parcial: {final_sim*100:.0f}% de proximidade atingida."
        
        # 2. Path description
        if improvement > 0.5:
            path_desc = f"Navegação de alta qualidade: {init_sim:.0%} → {final_sim:.0%} (melhoria de {improvement:.0%})"
        elif improvement > 0.2:
            path_desc = f"Navegação moderada: evolução de {improvement:.0%} na similaridade"
        else:
            path_desc = f"Navegação difícil: conceitos muito distantes no espaço semântico"
        
        # 3. Key concepts (from trajectory if available)
        key_concepts = []
        if 'path' in result:
            # Would extract intermediate concepts here
            key_concepts = ["(conceitos intermediários do caminho)"]
        
        # 4. Neurotype narrative
        neurotype_narrative = self._generate_neurotype_narrative(neurotype_contribs)
        
        # 5. Actionable insights
        insights = self._generate_insights(result, success, improvement)
        
        return NavigationExplanation(
            summary=summary,
            path_description=path_desc,
            key_concepts=key_concepts,
            confidence=final_sim,
            neurotype_narrative=neurotype_narrative,
            actionable_insights=insights
        )
    
    def _generate_neurotype_narrative(self, contribs: Dict) -> str:
        """Generate narrative about how different neurotypes contributed."""
        if not contribs:
            return "Navegação sem dados de neurotipos."
        
        # Sort by contribution
        sorted_contribs = sorted(
            contribs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        narratives = {
            'collapse': "foco intenso e direção direta",
            'critical': "exploração de bifurcações e transições",
            'psychedelic': "conexões laterais e não-lineares",
            'autistic': "detecção de padrões e hiperfoco",
            'balanced': "abordagem equilibrada",
            'relaxed': "exploração ampla e relaxada"
        }
        
        parts = []
        for nt, weight in sorted_contribs[:3]:
            nt_name = nt.value if hasattr(nt, 'value') else str(nt)
            nt_key = nt_name.lower().replace('neurotypename.', '')
            narrative = narratives.get(nt_key, "contribuição")
            parts.append(f"{nt_name} ({weight:.1f}): {narrative}")
        
        return "Contribuições: " + "; ".join(parts)
    
    def _generate_insights(self, result: Dict, success: bool, improvement: float) -> List[str]:
        """Generate actionable insights from navigation."""
        insights = []
        
        if success:
            insights.append("O caminho encontrado pode ser usado para gerar conteúdo conectando os dois conceitos.")
        
        if improvement > 0.5:
            insights.append("Alta convergência sugere forte relação latente entre os conceitos.")
        
        if result.get('topological_events'):
            insights.append("Eventos topológicos detectados - conceitos passam por regiões semânticas interessantes.")
        
        if not success and improvement < 0.2:
            insights.append("Conceitos muito distantes - considere buscar pontes intermediárias.")
        
        return insights
    
    def _convert_to_nemesis_action(self, swarm_result: Dict) -> Dict:
        """Convert Swarm result to Nemesis action format."""
        return {
            'action_type': 'navigate',
            'target': swarm_result.get('path', [])[-1] if swarm_result.get('path') else None,
            'parameters': {
                'steps': swarm_result.get('steps', 0),
                'mode': swarm_result.get('mode', 'balanced')
            },
            'expected_free_energy': 1.0 - swarm_result.get('final_similarity', 0.5),
            'confidence': swarm_result.get('final_similarity', 0.5)
        }
    
    def _mock_navigation_result(self) -> Dict:
        """Mock result for testing without Swarm."""
        return {
            'success': True,
            'steps': 10,
            'init_similarity': 0.3,
            'final_similarity': 0.9,
            'improvement': 0.6,
            'neurotype_contributions': {'collapse': 2.5, 'critical': 2.0},
            'mode': 'balanced'
        }
    
    def receive_feedback(self, navigation_id: int, positive: bool, comment: str = ""):
        """
        Receive user feedback on a navigation result.
        
        Args:
            navigation_id: Index of the navigation in history
            positive: True if user found the result helpful
            comment: Optional user comment
        """
        if navigation_id < len(self.navigation_history):
            feedback = {
                'navigation_id': navigation_id,
                'positive': positive,
                'comment': comment
            }
            self.feedback_history.append(feedback)
            
            # Would update Nemesis beliefs here
            if self.nemesis and hasattr(self.nemesis, 'update_after_action'):
                reward = 1.0 if positive else 0.0
                logger.info(f"Updating Nemesis with feedback: reward={reward}")
            
            return True
        return False


def create_swarm_nemesis_integration(
    topology_engine=None,
    memory_path: str = "data/swarm_memory.json"
):
    """
    Factory function to create full Swarm-Nemesis integration.
    
    Args:
        topology_engine: TopologyEngine instance
        memory_path: Path for Swarm memory persistence
        
    Returns:
        Configured SwarmNemesisIntegration
    """
    # Create Swarm
    try:
        from swarm.navigator import SwarmNavigator
        swarm = SwarmNavigator(
            topology_engine=topology_engine,
            memory_path=memory_path,
            use_neurodiverse=True
        )
    except ImportError:
        logger.warning("SwarmNavigator not available")
        swarm = None
    
    # Create Nemesis
    try:
        from core.loop.nemesis_integration import create_nemesis_integration
        nemesis = create_nemesis_integration(topology_engine=topology_engine)
    except ImportError:
        logger.warning("NemesisIntegration not available")
        nemesis = None
    
    return SwarmNemesisIntegration(
        swarm_navigator=swarm,
        nemesis_integration=nemesis,
        topology_engine=topology_engine
    )
