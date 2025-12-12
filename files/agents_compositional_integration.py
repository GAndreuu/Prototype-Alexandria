"""
Agents ↔ Compositional Integration
===================================

Conecta os agentes (Action, Bridge, Critic, Oracle) ao raciocínio composicional.

Isso permite:
- ActionAgent escolher ações via caminhos geodésicos
- BridgeAgent traduzir entre representações geometricamente
- CriticAgent avaliar usando energia do caminho
- NeuralOracle sintetizar seguindo geodésicas

Teoria:
    action = argmin_a G(a) onde G usa distância geodésica
    
    synthesis = ∫ transform(γ(t)) dt ao longo da geodésica
    
    critique = F(path) + complexity(path)

Uso:
    from agents_compositional_integration import AgentsCompositionalIntegration
    
    aci = AgentsCompositionalIntegration(bridge, compositional, agents)
    
    # Agente escolhe ação geometricamente
    action = aci.action_geometric(state, goals)
    
    # Oracle sintetiza via geodésica
    synthesis = aci.oracle_synthesis_geometric(sources, target)

Autor: G (Alexandria Project)
Versão: 1.0
Fase: 3.1 - Agentes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class AgentsCompositionalConfig:
    """Configuração da integração Agents ↔ Compositional."""
    
    # ActionAgent
    action_use_geodesic: bool = True
    action_samples: int = 10
    action_horizon: int = 3
    
    # BridgeAgent  
    bridge_interpolation_steps: int = 20
    bridge_preserve_structure: bool = True
    
    # CriticAgent
    critic_energy_weight: float = 0.5
    critic_complexity_weight: float = 0.3
    critic_coherence_weight: float = 0.2
    
    # NeuralOracle
    oracle_synthesis_steps: int = 50
    oracle_residual_accumulation: bool = True
    oracle_attractor_guidance: bool = True


# =============================================================================
# GEOMETRIC ACTION AGENT
# =============================================================================

@dataclass
class GeometricActionResult:
    """Resultado de seleção de ação geométrica."""
    action_type: str
    target_state: np.ndarray
    geodesic_path: np.ndarray
    expected_reward: float
    path_energy: float
    composition_trace: List[str]
    confidence: float


class GeometricActionAgent:
    """
    Action Agent que seleciona ações via caminhos geodésicos.
    
    Em vez de avaliar ações em espaço euclidiano, considera
    a geometria do manifold para escolher trajetos.
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        base_agent=None,
        config: Optional[AgentsCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.base_agent = base_agent
        self.config = config or AgentsCompositionalConfig()
    
    def select_action(
        self,
        current_state: np.ndarray,
        goals: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> GeometricActionResult:
        """
        Seleciona ação considerando caminhos geodésicos para cada goal.
        """
        # Projetar para manifold
        if hasattr(self.bridge, '_project_to_latent'):
            current = self.bridge._project_to_latent(current_state)
        else:
            current = current_state
        
        best_result = None
        best_score = float('-inf')
        
        for goal in goals:
            # Projetar goal
            if hasattr(self.bridge, '_project_to_latent'):
                goal_proj = self.bridge._project_to_latent(goal)
            else:
                goal_proj = goal
            
            # Computar caminho geodésico
            if self.compositional is not None:
                try:
                    path_result = self.compositional.reason(current, goal_proj)
                    path = path_result.points
                    trace = path_result.composition_trace
                    path_energy = self._compute_path_energy(path)
                except:
                    path = np.array([current, goal_proj])
                    trace = ["direct"]
                    path_energy = np.linalg.norm(goal_proj - current)
            else:
                path = np.array([current, goal_proj])
                trace = ["direct"]
                path_energy = np.linalg.norm(goal_proj - current)
            
            # Avaliar ação
            expected_reward = self._estimate_reward(goal_proj, context)
            score = expected_reward - 0.3 * path_energy
            
            if score > best_score:
                best_score = score
                best_result = GeometricActionResult(
                    action_type="move_to_goal",
                    target_state=goal_proj,
                    geodesic_path=path,
                    expected_reward=expected_reward,
                    path_energy=path_energy,
                    composition_trace=trace,
                    confidence=1.0 / (1.0 + path_energy)
                )
        
        return best_result or GeometricActionResult(
            action_type="explore",
            target_state=current,
            geodesic_path=np.array([current]),
            expected_reward=0.0,
            path_energy=0.0,
            composition_trace=["none"],
            confidence=0.5
        )
    
    def _compute_path_energy(self, path: np.ndarray) -> float:
        """Energia média ao longo do caminho."""
        if self.bridge is None or len(path) == 0:
            return 0.0
        
        energies = []
        for point in path[::max(1, len(path)//5)]:
            try:
                nearest = self.bridge.get_nearest_anchors(point, k=1)
                if nearest:
                    _, dist = nearest[0]
                    energies.append(dist)
            except:
                pass
        
        return float(np.mean(energies)) if energies else 0.0
    
    def _estimate_reward(
        self,
        target: np.ndarray,
        context: Optional[Dict]
    ) -> float:
        """Estima recompensa de atingir target."""
        # Baseado em proximidade a atratores (estados estáveis)
        if self.bridge is None:
            return 0.5
        
        try:
            nearest = self.bridge.get_nearest_anchors(target, k=1)
            if nearest:
                _, dist = nearest[0]
                return 1.0 / (1.0 + dist)
        except:
            pass
        
        return 0.5


# =============================================================================
# GEOMETRIC BRIDGE AGENT
# =============================================================================

@dataclass
class GeometricTranslation:
    """Tradução entre representações via geodésica."""
    source: np.ndarray
    target: np.ndarray
    intermediate_path: np.ndarray
    translation_fidelity: float
    structure_preserved: float
    composition_trace: List[str]


class GeometricBridgeAgent:
    """
    Bridge Agent que traduz entre representações via geodésicas.
    
    A tradução preserva estrutura seguindo o manifold em vez
    de interpolação linear.
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        base_agent=None,
        config: Optional[AgentsCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.base_agent = base_agent
        self.config = config or AgentsCompositionalConfig()
    
    def translate(
        self,
        source: np.ndarray,
        target_space: str = "latent",
        preserve_semantics: bool = True
    ) -> GeometricTranslation:
        """
        Traduz representação seguindo geodésica.
        """
        # Projetar source
        if hasattr(self.bridge, '_project_to_latent'):
            source_proj = self.bridge._project_to_latent(source)
        else:
            source_proj = source
        
        # Encontrar ponto alvo na nova representação
        if target_space == "latent":
            target = source_proj  # Já está no latent
        elif target_space == "quantized":
            # Encontrar código mais próximo
            try:
                point = self.bridge.embed(source)
                target = point.coordinates
            except:
                target = source_proj
        else:
            target = source_proj
        
        # Computar caminho de tradução
        n_steps = self.config.bridge_interpolation_steps
        
        if self.compositional is not None and preserve_semantics:
            try:
                result = self.compositional.reason(source_proj, target)
                path = result.points
                trace = result.composition_trace
            except:
                path = self._linear_path(source_proj, target, n_steps)
                trace = [f"step_{i}" for i in range(n_steps)]
        else:
            path = self._linear_path(source_proj, target, n_steps)
            trace = [f"step_{i}" for i in range(n_steps)]
        
        # Medir fidelidade
        fidelity = 1.0 / (1.0 + np.linalg.norm(target - source_proj))
        
        # Medir preservação de estrutura
        structure = self._measure_structure_preservation(path)
        
        return GeometricTranslation(
            source=source_proj,
            target=target,
            intermediate_path=path,
            translation_fidelity=fidelity,
            structure_preserved=structure,
            composition_trace=trace
        )
    
    def _linear_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """Caminho linear (fallback)."""
        path = np.zeros((steps, len(start)))
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            path[i] = start + t * (end - start)
        return path
    
    def _measure_structure_preservation(self, path: np.ndarray) -> float:
        """Mede quão bem a estrutura é preservada."""
        if self.bridge is None or len(path) < 2:
            return 0.5
        
        # Verificar se passa por atratores
        attractor_hits = 0
        for point in path[::max(1, len(path)//5)]:
            try:
                nearest = self.bridge.get_nearest_anchors(point, k=1)
                if nearest and nearest[0][1] < 0.3:
                    attractor_hits += 1
            except:
                pass
        
        return min(1.0, attractor_hits * 0.2)


# =============================================================================
# GEOMETRIC CRITIC AGENT
# =============================================================================

@dataclass
class GeometricCritique:
    """Crítica geométrica de uma saída."""
    overall_score: float
    energy_score: float
    complexity_score: float
    coherence_score: float
    path_analysis: Dict[str, float]
    suggestions: List[str]


class GeometricCriticAgent:
    """
    Critic Agent que avalia usando métricas geométricas.
    
    Considera:
    - Energia do caminho até o output
    - Complexidade (comprimento geodésico)
    - Coerência (consistência com atratores)
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        base_agent=None,
        config: Optional[AgentsCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.base_agent = base_agent
        self.config = config or AgentsCompositionalConfig()
    
    def critique(
        self,
        output: np.ndarray,
        context: Optional[np.ndarray] = None,
        path: Optional[np.ndarray] = None
    ) -> GeometricCritique:
        """
        Avalia output geometricamente.
        """
        # Projetar para manifold
        if hasattr(self.bridge, '_project_to_latent'):
            output_proj = self.bridge._project_to_latent(output)
        else:
            output_proj = output
        
        # Energia no ponto de output
        energy_score = self._compute_energy_score(output_proj)
        
        # Complexidade do caminho
        complexity_score = self._compute_complexity_score(path)
        
        # Coerência com estrutura
        coherence_score = self._compute_coherence_score(output_proj)
        
        # Score geral
        cfg = self.config
        overall = (
            cfg.critic_energy_weight * energy_score +
            cfg.critic_complexity_weight * complexity_score +
            cfg.critic_coherence_weight * coherence_score
        )
        
        # Análise do caminho
        path_analysis = {}
        if path is not None and len(path) > 0:
            path_analysis['length'] = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            path_analysis['curvature'] = self._compute_path_curvature(path)
            path_analysis['attractors_visited'] = self._count_attractors(path)
        
        # Sugestões
        suggestions = []
        if energy_score < 0.5:
            suggestions.append("Output está em região de alta energia - considere mover para atrator mais próximo")
        if complexity_score < 0.5:
            suggestions.append("Caminho muito complexo - busque rota mais direta")
        if coherence_score < 0.5:
            suggestions.append("Output não alinhado com estrutura - verifique consistência semântica")
        
        return GeometricCritique(
            overall_score=overall,
            energy_score=energy_score,
            complexity_score=complexity_score,
            coherence_score=coherence_score,
            path_analysis=path_analysis,
            suggestions=suggestions
        )
    
    def _compute_energy_score(self, point: np.ndarray) -> float:
        """Score baseado em energia (menor = melhor)."""
        if self.bridge is None:
            return 0.5
        
        try:
            nearest = self.bridge.get_nearest_anchors(point, k=1)
            if nearest:
                _, dist = nearest[0]
                return 1.0 / (1.0 + dist)
        except:
            pass
        
        return 0.5
    
    def _compute_complexity_score(self, path: Optional[np.ndarray]) -> float:
        """Score baseado em complexidade (menor = melhor)."""
        if path is None or len(path) < 2:
            return 1.0
        
        # Comprimento total
        total_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        
        # Distância direta
        direct_dist = np.linalg.norm(path[-1] - path[0])
        
        # Razão (1 = caminho direto)
        if direct_dist > 1e-8:
            ratio = direct_dist / (total_length + 1e-8)
        else:
            ratio = 1.0
        
        return min(1.0, ratio)
    
    def _compute_coherence_score(self, point: np.ndarray) -> float:
        """Score baseado em coerência com estrutura."""
        if self.bridge is None:
            return 0.5
        
        try:
            # Verificar se está perto de atrator
            nearest = self.bridge.get_nearest_anchors(point, k=4)
            if nearest:
                # Média das distâncias aos 4 mais próximos
                avg_dist = np.mean([d for _, d in nearest])
                return 1.0 / (1.0 + avg_dist)
        except:
            pass
        
        return 0.5
    
    def _compute_path_curvature(self, path: np.ndarray) -> float:
        """Curvatura total do caminho."""
        if self.bridge is None or len(path) < 3:
            return 0.0
        
        total = 0.0
        for point in path[::max(1, len(path)//5)]:
            try:
                g = self.bridge.compute_metric_deformation(point)
                total += max(0, np.trace(g) - len(point))
            except:
                pass
        
        return total
    
    def _count_attractors(self, path: np.ndarray) -> int:
        """Conta atratores visitados."""
        if self.bridge is None:
            return 0
        
        count = 0
        visited = set()
        
        for point in path:
            try:
                nearest = self.bridge.get_nearest_anchors(point, k=1)
                if nearest:
                    anchor, dist = nearest[0]
                    if dist < 0.3 and anchor.global_idx not in visited:
                        count += 1
                        visited.add(anchor.global_idx)
            except:
                pass
        
        return count


# =============================================================================
# GEOMETRIC ORACLE
# =============================================================================

@dataclass
class GeometricSynthesis:
    """Síntese geométrica do Oracle."""
    synthesized: np.ndarray
    sources_used: List[np.ndarray]
    geodesic_paths: List[np.ndarray]
    accumulated_residual: np.ndarray
    composition_trace: List[str]
    synthesis_quality: float


class GeometricNeuralOracle:
    """
    Neural Oracle que sintetiza seguindo geodésicas.
    
    A síntese é feita acumulando transformações ao longo
    de caminhos geodésicos de múltiplas fontes.
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        base_oracle=None,
        config: Optional[AgentsCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.base_oracle = base_oracle
        self.config = config or AgentsCompositionalConfig()
    
    def synthesize(
        self,
        sources: List[np.ndarray],
        target_direction: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None
    ) -> GeometricSynthesis:
        """
        Sintetiza a partir de múltiplas fontes via geodésicas.
        """
        if weights is None:
            weights = [1.0 / len(sources)] * len(sources)
        
        # Projetar fontes
        sources_proj = []
        for src in sources:
            if hasattr(self.bridge, '_project_to_latent'):
                sources_proj.append(self.bridge._project_to_latent(src))
            else:
                sources_proj.append(src)
        
        # Encontrar centroide como ponto inicial
        centroid = np.average(sources_proj, axis=0, weights=weights)
        
        # Acumular resíduos de cada fonte
        accumulated = np.zeros_like(centroid)
        all_paths = []
        all_traces = []
        
        for src, w in zip(sources_proj, weights):
            if self.compositional is not None:
                try:
                    result = self.compositional.reason(centroid, src)
                    path = result.points
                    residual = result.cumulative_vector - centroid
                    trace = result.composition_trace
                except:
                    path = np.array([centroid, src])
                    residual = src - centroid
                    trace = ["direct"]
            else:
                path = np.array([centroid, src])
                residual = src - centroid
                trace = ["direct"]
            
            accumulated += w * residual
            all_paths.append(path)
            all_traces.extend(trace)
        
        # Resultado final
        synthesized = centroid + accumulated
        
        # Aplicar direção target se especificada
        if target_direction is not None:
            if hasattr(self.bridge, '_project_to_latent'):
                target_direction = self.bridge._project_to_latent(target_direction)
            
            # Projetar na direção do target
            direction = target_direction - synthesized
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Mover parcialmente na direção
            synthesized = synthesized + 0.3 * direction * np.linalg.norm(accumulated)
        
        # Guiar para atrator mais próximo se configurado
        if self.config.oracle_attractor_guidance and self.bridge is not None:
            try:
                nearest = self.bridge.get_nearest_anchors(synthesized, k=1)
                if nearest:
                    anchor, dist = nearest[0]
                    if dist < 0.5:
                        # Pull toward attractor
                        synthesized = synthesized + 0.2 * (anchor.coordinates - synthesized)
            except:
                pass
        
        # Qualidade da síntese
        quality = self._compute_synthesis_quality(synthesized, sources_proj)
        
        return GeometricSynthesis(
            synthesized=synthesized,
            sources_used=sources_proj,
            geodesic_paths=all_paths,
            accumulated_residual=accumulated,
            composition_trace=all_traces[:10],  # Limitar tamanho
            synthesis_quality=quality
        )
    
    def _compute_synthesis_quality(
        self,
        synthesized: np.ndarray,
        sources: List[np.ndarray]
    ) -> float:
        """Avalia qualidade da síntese."""
        # Distância média às fontes (menor = melhor)
        avg_dist = np.mean([np.linalg.norm(synthesized - src) for src in sources])
        dist_score = 1.0 / (1.0 + avg_dist)
        
        # Proximidade a atrator (estrutura)
        structure_score = 0.5
        if self.bridge is not None:
            try:
                nearest = self.bridge.get_nearest_anchors(synthesized, k=1)
                if nearest:
                    _, d = nearest[0]
                    structure_score = 1.0 / (1.0 + d)
            except:
                pass
        
        return 0.5 * dist_score + 0.5 * structure_score


# =============================================================================
# INTEGRAÇÃO UNIFICADA
# =============================================================================

class AgentsCompositionalIntegration:
    """
    Integração unificada de todos os agentes com raciocínio composicional.
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        action_agent=None,
        bridge_agent=None,
        critic_agent=None,
        oracle=None,
        config: Optional[AgentsCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.compositional = compositional
        self.config = config or AgentsCompositionalConfig()
        
        # Wrappers geométricos
        self.geometric_action = GeometricActionAgent(
            bridge, compositional, action_agent, self.config
        )
        self.geometric_bridge = GeometricBridgeAgent(
            bridge, compositional, bridge_agent, self.config
        )
        self.geometric_critic = GeometricCriticAgent(
            bridge, compositional, critic_agent, self.config
        )
        self.geometric_oracle = GeometricNeuralOracle(
            bridge, compositional, oracle, self.config
        )
    
    def action_geometric(
        self,
        state: np.ndarray,
        goals: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> GeometricActionResult:
        """Seleciona ação via agente geométrico."""
        return self.geometric_action.select_action(state, goals, context)
    
    def translate_geometric(
        self,
        source: np.ndarray,
        target_space: str = "latent"
    ) -> GeometricTranslation:
        """Traduz representação geometricamente."""
        return self.geometric_bridge.translate(source, target_space)
    
    def critique_geometric(
        self,
        output: np.ndarray,
        context: Optional[np.ndarray] = None,
        path: Optional[np.ndarray] = None
    ) -> GeometricCritique:
        """Avalia output geometricamente."""
        return self.geometric_critic.critique(output, context, path)
    
    def synthesize_geometric(
        self,
        sources: List[np.ndarray],
        target_direction: Optional[np.ndarray] = None
    ) -> GeometricSynthesis:
        """Sintetiza a partir de fontes geometricamente."""
        return self.geometric_oracle.synthesize(sources, target_direction)
    
    def full_pipeline(
        self,
        query: np.ndarray,
        candidates: List[np.ndarray],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Pipeline completo:
        1. Seleciona melhor candidato (Action)
        2. Traduz para representação final (Bridge)
        3. Critica resultado (Critic)
        4. Sintetiza versão melhorada (Oracle)
        """
        results = {}
        
        # 1. Selecionar
        action_result = self.action_geometric(query, candidates, context)
        results['action'] = action_result
        
        # 2. Traduzir
        translation = self.translate_geometric(action_result.target_state)
        results['translation'] = translation
        
        # 3. Criticar
        critique = self.critique_geometric(
            translation.target,
            query,
            action_result.geodesic_path
        )
        results['critique'] = critique
        
        # 4. Sintetizar (se crítica baixa)
        if critique.overall_score < 0.7:
            synthesis = self.synthesize_geometric(
                [query, action_result.target_state],
                target_direction=translation.target
            )
            results['synthesis'] = synthesis
            results['final_output'] = synthesis.synthesized
        else:
            results['final_output'] = translation.target
        
        return results
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "has_compositional": self.compositional is not None,
            "config": {
                "action_use_geodesic": self.config.action_use_geodesic,
                "oracle_residual_accumulation": self.config.oracle_residual_accumulation
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_agents_compositional(
    bridge,
    compositional=None,
    **config_kwargs
) -> AgentsCompositionalIntegration:
    """Factory function."""
    config = AgentsCompositionalConfig(**config_kwargs)
    return AgentsCompositionalIntegration(bridge, compositional, config=config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agents ↔ Compositional Integration - Teste")
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
    print("\n2. Criando integração Agents ↔ Compositional...")
    aci = AgentsCompositionalIntegration(bridge)
    
    # Teste: Action Agent
    print("\n3. Testando Action Agent geométrico...")
    state = np.random.randn(512).astype(np.float32)
    goals = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    
    action_result = aci.action_geometric(state, goals)
    print(f"   Action: {action_result.action_type}")
    print(f"   Path energy: {action_result.path_energy:.4f}")
    print(f"   Confidence: {action_result.confidence:.4f}")
    
    # Teste: Bridge Agent
    print("\n4. Testando Bridge Agent geométrico...")
    translation = aci.translate_geometric(state)
    print(f"   Fidelity: {translation.translation_fidelity:.4f}")
    print(f"   Structure preserved: {translation.structure_preserved:.4f}")
    
    # Teste: Critic Agent
    print("\n5. Testando Critic Agent geométrico...")
    critique = aci.critique_geometric(state, path=action_result.geodesic_path)
    print(f"   Overall: {critique.overall_score:.4f}")
    print(f"   Energy: {critique.energy_score:.4f}")
    print(f"   Coherence: {critique.coherence_score:.4f}")
    if critique.suggestions:
        print(f"   Sugestões: {critique.suggestions[0][:50]}...")
    
    # Teste: Oracle
    print("\n6. Testando Neural Oracle geométrico...")
    sources = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    synthesis = aci.synthesize_geometric(sources)
    print(f"   Quality: {synthesis.synthesis_quality:.4f}")
    print(f"   Trace: {synthesis.composition_trace[:3]}")
    
    # Teste: Pipeline completo
    print("\n7. Testando pipeline completo...")
    pipeline_result = aci.full_pipeline(state, goals)
    print(f"   Action confidence: {pipeline_result['action'].confidence:.4f}")
    print(f"   Critique score: {pipeline_result['critique'].overall_score:.4f}")
    print(f"   Has synthesis: {'synthesis' in pipeline_result}")
    
    # Stats
    print("\n8. Estatísticas:")
    stats = aci.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
