"""
Nemesis ↔ Bridge Integration
=============================

Conecta o sistema de Active Inference (Nemesis) ao manifold curvo (Bridge).

O Expected Free Energy (EFE) agora é calculado considerando:
- Distância geodésica (não euclidiana)
- Curvatura local (atratores VQ-VAE)
- Energia livre do campo

Isso permite que o agente "sinta" a topologia semântica ao planejar ações.

Teoria:
    G(π) = Risk + Ambiguity
    
    Onde agora:
    - Risk = D_geodesic[Q(o|π) || P(o)]  (distância no manifold)
    - Ambiguity = ∫ H(x) · κ(x) dx  (entropia ponderada por curvatura)

Uso:
    from nemesis_bridge_integration import NemesisBridgeIntegration
    
    nbi = NemesisBridgeIntegration(nemesis, bridge)
    action = nbi.select_action_geometric(gaps, hypotheses)

Autor: G (Alexandria Project)
Versão: 1.0
Fase: 1.1 - Fundação
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class NemesisBridgeConfig:
    """Configuração da integração Nemesis ↔ Bridge."""
    
    # Pesos do EFE
    risk_weight: float = 1.0
    ambiguity_weight: float = 1.0
    curvature_weight: float = 0.3       # Peso da curvatura no cálculo
    
    # Geodésica
    use_geodesic_distance: bool = True  # Usar distância geodésica vs euclidiana
    geodesic_samples: int = 10          # Amostras ao longo da geodésica
    
    # Campo
    use_field_energy: bool = True       # Incorporar F do campo
    field_temperature: float = 1.0      # Temperatura para exploração
    
    # Planning
    planning_horizon: int = 5
    num_action_samples: int = 20
    
    # Curvatura
    curvature_bonus: float = 0.2        # Bônus para regiões de alta curvatura
    attractor_bonus: float = 0.3        # Bônus para proximidade de atratores


# =============================================================================
# MÉTRICAS GEOMÉTRICAS
# =============================================================================

@dataclass
class GeometricEFE:
    """Expected Free Energy calculado geometricamente."""
    total: float
    risk: float
    ambiguity: float
    curvature_term: float
    field_energy: float
    geodesic_length: float
    attractors_visited: int
    
    def __repr__(self):
        return (f"GeometricEFE(G={self.total:.4f}, "
                f"risk={self.risk:.4f}, amb={self.ambiguity:.4f})")


@dataclass
class GeometricAction:
    """Ação enriquecida com informação geométrica."""
    action_type: str
    target: str
    parameters: Dict[str, Any]
    
    # EFE geométrico
    geometric_efe: GeometricEFE
    
    # Caminho no manifold
    geodesic_path: Optional[np.ndarray] = None
    composition_trace: List[str] = field(default_factory=list)
    
    # Scores
    epistemic_value: float = 0.0        # Valor de informação
    pragmatic_value: float = 0.0        # Valor prático
    geometric_value: float = 0.0        # Valor pela geometria
    
    @property
    def total_value(self) -> float:
        return self.epistemic_value + self.pragmatic_value + self.geometric_value
    
    @property
    def priority(self) -> float:
        return -self.geometric_efe.total + self.total_value


# =============================================================================
# INTEGRAÇÃO PRINCIPAL
# =============================================================================

class NemesisBridgeIntegration:
    """
    Integra Nemesis (Active Inference) com Bridge (Manifold Curvo).
    
    O agente agora planeja considerando a geometria do espaço semântico.
    """
    
    def __init__(
        self,
        bridge,                          # VQVAEManifoldBridge
        nemesis=None,                    # NemesisIntegration (opcional)
        compositional=None,              # CompositionalReasoner (opcional)
        config: Optional[NemesisBridgeConfig] = None
    ):
        self.bridge = bridge
        self.nemesis = nemesis
        self.compositional = compositional
        self.config = config or NemesisBridgeConfig()
        
        # Estado
        self._current_belief = None
        self._action_history = []
        
        logger.info("NemesisBridgeIntegration initialized")
    
    # =========================================================================
    # API PRINCIPAL
    # =========================================================================
    
    def select_action_geometric(
        self,
        gaps: List[Dict],
        hypotheses: List[Dict],
        current_state: Optional[np.ndarray] = None
    ) -> GeometricAction:
        """
        Seleciona melhor ação usando EFE geométrico.
        
        Args:
            gaps: Lista de knowledge gaps detectados
            hypotheses: Lista de hipóteses do AbductionEngine
            current_state: Estado atual (embedding)
        
        Returns:
            GeometricAction com maior prioridade
        """
        if current_state is None:
            current_state = self._get_current_state()
        
        # Gerar ações candidatas
        candidates = self._generate_candidates(gaps, hypotheses, current_state)
        
        # Calcular EFE geométrico para cada
        for action in candidates:
            action.geometric_efe = self._compute_geometric_efe(
                current_state, action
            )
            action.geometric_value = self._compute_geometric_value(action)
        
        # Selecionar melhor (menor EFE, maior valor)
        best = max(candidates, key=lambda a: a.priority)
        
        # Registrar
        self._action_history.append(best)
        
        return best
    
    def update_beliefs_geometric(
        self,
        observation: np.ndarray,
        action: GeometricAction,
        reward: float
    ) -> Dict[str, float]:
        """
        Atualiza beliefs considerando geometria.
        
        Args:
            observation: Embedding observado após ação
            action: Ação que foi executada
            reward: Sinal de recompensa
        
        Returns:
            Métricas da atualização
        """
        metrics = {}
        
        # Projetar observação no manifold
        if hasattr(self.bridge, 'embed'):
            obs_point = self.bridge.embed(observation)
            metrics['projection_distance'] = obs_point.nearest_anchor_distance
        
        # Prediction error no espaço curvo
        if action.geodesic_path is not None and len(action.geodesic_path) > 0:
            expected_end = action.geodesic_path[-1]
            if hasattr(self.bridge, '_project_to_latent'):
                obs_proj = self.bridge._project_to_latent(observation)
            else:
                obs_proj = observation
            
            # Erro = distância geodésica entre esperado e observado
            prediction_error = self._geodesic_distance(expected_end, obs_proj)
            metrics['prediction_error'] = prediction_error
        
        # Free energy variacional
        if self.config.use_field_energy:
            F = self._compute_field_energy(observation)
            metrics['free_energy'] = F
        
        # Atualizar Nemesis se disponível
        if self.nemesis is not None:
            try:
                self.nemesis.update_after_action(action, observation, reward)
                metrics['nemesis_updated'] = True
            except Exception as e:
                logger.warning(f"Nemesis update failed: {e}")
                metrics['nemesis_updated'] = False
        
        # Atualizar belief interno
        self._current_belief = observation
        
        return metrics
    
    # =========================================================================
    # CÁLCULO DE EFE GEOMÉTRICO
    # =========================================================================
    
    def _compute_geometric_efe(
        self,
        current_state: np.ndarray,
        action: GeometricAction
    ) -> GeometricEFE:
        """
        Calcula Expected Free Energy no manifold curvo.
        
        G(π) = Risk + Ambiguity + Curvature_term
        """
        # Obter target da ação
        target = self._action_to_target(action, current_state)
        
        if target is None:
            return GeometricEFE(
                total=float('inf'),
                risk=float('inf'),
                ambiguity=float('inf'),
                curvature_term=0.0,
                field_energy=0.0,
                geodesic_length=0.0,
                attractors_visited=0
            )
        
        # Computar geodésica
        geodesic_length, path, attractors = self._compute_geodesic_info(
            current_state, target
        )
        action.geodesic_path = path
        
        # Risk: distância geodésica ao target preferido
        risk = self._compute_risk_geometric(current_state, target, geodesic_length)
        
        # Ambiguity: incerteza ao longo do caminho
        ambiguity = self._compute_ambiguity_geometric(path)
        
        # Termo de curvatura (bônus/penalidade)
        curvature_term = self._compute_curvature_term(path)
        
        # Energia do campo
        field_energy = 0.0
        if self.config.use_field_energy:
            field_energy = self._compute_field_energy(target)
        
        # Combinar
        total = (
            self.config.risk_weight * risk +
            self.config.ambiguity_weight * ambiguity -
            self.config.curvature_weight * curvature_term +
            field_energy / self.config.field_temperature
        )
        
        return GeometricEFE(
            total=total,
            risk=risk,
            ambiguity=ambiguity,
            curvature_term=curvature_term,
            field_energy=field_energy,
            geodesic_length=geodesic_length,
            attractors_visited=attractors
        )
    
    def _compute_risk_geometric(
        self,
        current: np.ndarray,
        target: np.ndarray,
        geodesic_length: float
    ) -> float:
        """
        Risk = quão longe do target preferido (geodesicamente).
        """
        if self.config.use_geodesic_distance:
            return geodesic_length
        else:
            # Fallback: euclidiana
            return np.linalg.norm(target - current)
    
    def _compute_ambiguity_geometric(
        self,
        path: Optional[np.ndarray]
    ) -> float:
        """
        Ambiguity = incerteza média ao longo do caminho.
        
        Medida pela distância aos atratores mais próximos.
        """
        if path is None or len(path) == 0:
            return 1.0
        
        uncertainties = []
        
        for point in path[::max(1, len(path)//self.config.geodesic_samples)]:
            # Incerteza = distância ao atrator mais próximo
            if self.bridge is not None:
                try:
                    nearest = self.bridge.get_nearest_anchors(point, k=1)
                    if nearest:
                        _, dist = nearest[0]
                        uncertainties.append(dist)
                except:
                    uncertainties.append(1.0)
            else:
                uncertainties.append(1.0)
        
        return float(np.mean(uncertainties)) if uncertainties else 1.0
    
    def _compute_curvature_term(
        self,
        path: Optional[np.ndarray]
    ) -> float:
        """
        Termo de curvatura: regiões curvas são mais "interessantes".
        
        Bônus para caminhos que passam por regiões de alta curvatura.
        """
        if path is None or len(path) == 0 or self.bridge is None:
            return 0.0
        
        curvatures = []
        
        for point in path[::max(1, len(path)//5)]:  # Amostrar 5 pontos
            try:
                # Curvatura ≈ traço da métrica - dim (desvio do flat)
                g = self.bridge.compute_metric_deformation(point)
                curvature = np.trace(g) - len(point)
                curvatures.append(max(0, curvature))
            except:
                pass
        
        return float(np.mean(curvatures)) if curvatures else 0.0
    
    def _compute_field_energy(
        self,
        point: np.ndarray
    ) -> float:
        """
        Energia livre do campo no ponto.
        """
        if self.bridge is None:
            return 0.0
        
        try:
            # F ≈ distância média aos atratores
            if hasattr(self.bridge, '_project_to_latent'):
                point = self.bridge._project_to_latent(point)
            
            nearest = self.bridge.get_nearest_anchors(point, k=4)
            if nearest:
                return float(np.mean([d for _, d in nearest]))
        except:
            pass
        
        return 0.0
    
    # =========================================================================
    # VALOR GEOMÉTRICO
    # =========================================================================
    
    def _compute_geometric_value(
        self,
        action: GeometricAction
    ) -> float:
        """
        Valor adicional pela geometria do caminho.
        
        - Bônus por passar perto de atratores
        - Bônus por regiões de alta curvatura (novidade estrutural)
        """
        value = 0.0
        
        efe = action.geometric_efe
        
        # Bônus por atratores visitados
        value += self.config.attractor_bonus * efe.attractors_visited
        
        # Bônus por curvatura (explorar regiões estruturadas)
        value += self.config.curvature_bonus * efe.curvature_term
        
        # Penalidade por energia alta (estados improváveis)
        value -= 0.1 * efe.field_energy
        
        return value
    
    # =========================================================================
    # GEODÉSICA
    # =========================================================================
    
    def _compute_geodesic_info(
        self,
        start: np.ndarray,
        end: np.ndarray
    ) -> Tuple[float, Optional[np.ndarray], int]:
        """
        Computa informações geodésicas entre dois pontos.
        
        Returns:
            (comprimento, caminho, atratores_visitados)
        """
        # Projetar para espaço do manifold
        if hasattr(self.bridge, '_project_to_latent'):
            start = self.bridge._project_to_latent(start)
            end = self.bridge._project_to_latent(end)
        
        # Usar compositional se disponível
        if self.compositional is not None:
            try:
                result = self.compositional.reason(start, end)
                return (
                    result.path_length,
                    result.points,
                    len([s for s in result.steps if s.free_energy < 0.5])
                )
            except Exception as e:
                logger.debug(f"Compositional failed: {e}")
        
        # Fallback: interpolação linear
        n_steps = 10
        path = np.zeros((n_steps, len(start)))
        for i in range(n_steps):
            t = i / (n_steps - 1)
            path[i] = start + t * (end - start)
        
        length = np.linalg.norm(end - start)
        
        # Contar atratores próximos ao caminho
        attractors = 0
        if self.bridge is not None:
            for point in path:
                try:
                    nearest = self.bridge.get_nearest_anchors(point, k=1)
                    if nearest and nearest[0][1] < 0.3:
                        attractors += 1
                except:
                    pass
        
        return length, path, attractors
    
    def _geodesic_distance(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Distância geodésica entre dois pontos.
        """
        length, _, _ = self._compute_geodesic_info(a, b)
        return length
    
    # =========================================================================
    # GERAÇÃO DE CANDIDATOS
    # =========================================================================
    
    def _generate_candidates(
        self,
        gaps: List[Dict],
        hypotheses: List[Dict],
        current_state: np.ndarray
    ) -> List[GeometricAction]:
        """
        Gera ações candidatas a partir de gaps e hipóteses.
        """
        candidates = []
        
        # De hipóteses
        for hyp in hypotheses[:self.config.num_action_samples]:
            action = GeometricAction(
                action_type="test_hypothesis",
                target=hyp.get('hypothesis_text', str(hyp)),
                parameters={
                    'hypothesis_id': hyp.get('id'),
                    'confidence': hyp.get('confidence_score', 0.5),
                    'source': hyp.get('source_cluster'),
                    'target': hyp.get('target_cluster')
                },
                geometric_efe=None,
                epistemic_value=hyp.get('confidence_score', 0.5),
                pragmatic_value=0.3
            )
            candidates.append(action)
        
        # De gaps
        for gap in gaps[:self.config.num_action_samples]:
            action = GeometricAction(
                action_type="fill_gap",
                target=gap.get('description', str(gap)),
                parameters={
                    'gap_id': gap.get('gap_id'),
                    'gap_type': gap.get('gap_type'),
                    'priority': gap.get('priority_score', 0.5)
                },
                geometric_efe=None,
                epistemic_value=gap.get('priority_score', 0.5) * 1.2,  # Gaps têm bônus epistêmico
                pragmatic_value=0.2
            )
            candidates.append(action)
        
        # Ação exploratória (sempre incluir)
        candidates.append(GeometricAction(
            action_type="explore",
            target="random_exploration",
            parameters={'temperature': self.config.field_temperature},
            geometric_efe=None,
            epistemic_value=0.8,  # Alta valor epistêmico
            pragmatic_value=0.1
        ))
        
        return candidates
    
    def _action_to_target(
        self,
        action: GeometricAction,
        current_state: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Converte ação em vetor target no manifold.
        """
        if action.action_type == "explore":
            # Target aleatório na direção de maior incerteza
            if self.bridge is not None:
                try:
                    # Encontrar direção de maior F (mais incerto)
                    direction = np.random.randn(len(current_state))
                    direction = direction / np.linalg.norm(direction)
                    return current_state + direction * 0.5
                except:
                    pass
            return current_state + np.random.randn(len(current_state)) * 0.3
        
        # Para outras ações, usar embedding do target se disponível
        params = action.parameters
        
        if 'target_embedding' in params:
            return np.array(params['target_embedding'])
        
        # Fallback: perturbação do estado atual
        return current_state + np.random.randn(len(current_state)) * 0.2
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def _get_current_state(self) -> np.ndarray:
        """Obtém estado atual do sistema."""
        if self._current_belief is not None:
            return self._current_belief
        
        if self.nemesis is not None and hasattr(self.nemesis, 'get_belief'):
            return self.nemesis.get_belief()
        
        # Fallback: vetor zero
        dim = 384
        if self.bridge is not None:
            dim = self.bridge.config.embedding_dim
        return np.zeros(dim)
    
    def set_belief(self, belief: np.ndarray):
        """Define belief atual."""
        self._current_belief = belief
    
    def get_action_history(self) -> List[GeometricAction]:
        """Retorna histórico de ações."""
        return self._action_history.copy()
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "has_nemesis": self.nemesis is not None,
            "has_compositional": self.compositional is not None,
            "action_history_size": len(self._action_history),
            "config": {
                "use_geodesic": self.config.use_geodesic_distance,
                "use_field": self.config.use_field_energy,
                "risk_weight": self.config.risk_weight,
                "ambiguity_weight": self.config.ambiguity_weight
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_nemesis_bridge(
    bridge,
    nemesis=None,
    compositional=None,
    **config_kwargs
) -> NemesisBridgeIntegration:
    """Factory function."""
    config = NemesisBridgeConfig(**config_kwargs)
    return NemesisBridgeIntegration(bridge, nemesis, compositional, config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Nemesis ↔ Bridge Integration - Teste")
    print("=" * 60)
    
    # Importar dependências
    try:
        from vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig, ProjectionMode
    except ImportError:
        print("ERRO: vqvae_manifold_bridge.py não encontrado")
        exit(1)
    
    # Mock VQ-VAE
    class MockVQVAE:
        def __init__(self):
            np.random.seed(42)
            self.quantizer = type('Q', (), {
                'codebooks': type('C', (), {
                    'detach': lambda: type('D', (), {
                        'cpu': lambda: type('N', (), {
                            'numpy': lambda: np.random.randn(4, 256, 128).astype(np.float32)
                        })()
                    })()
                })()
            })()
        def get_codebook(self):
            return self.quantizer.codebooks.detach().cpu().numpy()
    
    # Criar bridge
    print("\n1. Criando bridge...")
    bridge = VQVAEManifoldBridge(BridgeConfig(pull_strength=0.5))
    bridge.connect_vqvae(MockVQVAE())
    print(f"   Anchors: {len(bridge.anchor_points)}")
    
    # Criar integração
    print("\n2. Criando integração Nemesis ↔ Bridge...")
    nbi = NemesisBridgeIntegration(bridge)
    
    # Mock gaps e hypotheses
    gaps = [
        {'gap_id': 'g1', 'description': 'Missing link A-B', 'priority_score': 0.8},
        {'gap_id': 'g2', 'description': 'Orphaned cluster C', 'priority_score': 0.6},
    ]
    
    hypotheses = [
        {'id': 'h1', 'hypothesis_text': 'A causes B', 'confidence_score': 0.7},
        {'id': 'h2', 'hypothesis_text': 'C relates to D', 'confidence_score': 0.5},
    ]
    
    # Teste: selecionar ação
    print("\n3. Selecionando ação geométrica...")
    current = np.random.randn(384).astype(np.float32)
    
    action = nbi.select_action_geometric(gaps, hypotheses, current)
    
    print(f"   Ação: {action.action_type}")
    print(f"   Target: {action.target[:50]}...")
    print(f"   EFE: {action.geometric_efe}")
    print(f"   Priority: {action.priority:.4f}")
    
    # Teste: update beliefs
    print("\n4. Atualizando beliefs após observação...")
    observation = np.random.randn(384).astype(np.float32)
    
    metrics = nbi.update_beliefs_geometric(observation, action, reward=0.8)
    
    print(f"   Metrics: {metrics}")
    
    # Stats
    print("\n5. Estatísticas:")
    stats = nbi.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
