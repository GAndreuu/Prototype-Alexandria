"""
Compositional Geodesic Reasoning
================================

Raciocínio composicional via acumulação de resíduos durante travessia geodésica.

O vetor não apenas CHEGA ao destino — ele é TRANSFORMADO pelo caminho.
Cada nó visitado contribui um resíduo que codifica seu significado.

Teoria:
    v_out = v_in + ∫₀¹ R(γ(t), v(t)) dt
    
    Onde:
    - γ(t) é a geodésica no manifold curvo
    - R é a função de resíduo (Hebbiano, Attention, ou Campo)
    - A integral acumula transformações ao longo do caminho

Isso habilita:
    - Raciocínio em cadeia (A → B → C como composição)
    - Explicabilidade (o vetor codifica COMO chegou)
    - Analogia (mesmo caminho, diferentes pontos de partida)
    - Composição semântica ("rei - homem + mulher" via geodésica)

Uso:
    from compositional_reasoning import CompositionalReasoner
    
    reasoner = CompositionalReasoner(bridge, mycelial)
    
    result = reasoner.reason(query_embedding, target_embedding)
    print(f"Conceitos visitados: {result.composition_trace}")
    print(f"Vetor transformado shape: {result.cumulative_vector.shape}")

Autor: G (Alexandria Project)
Versão: 1.0
Status: S_λ → S_ι (estrutura emergente)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS E CONFIGURAÇÃO
# =============================================================================

class ResidualMode(Enum):
    """Modos de computação de resíduo."""
    HEBBIAN = "hebbian"          # Baseado em conexões Hebbianas
    ATTENTION = "attention"      # Mecanismo de atenção
    FIELD = "field"              # Gradiente de energia livre
    PROJECTION = "projection"    # Projeção no espaço tangente
    HYBRID = "hybrid"            # Combinação ponderada


class CompositionStrategy(Enum):
    """Estratégias de composição dos resíduos."""
    ADDITIVE = "additive"        # v = v + Σr (soma simples)
    GATED = "gated"              # v = v + gate(v) * Σr
    NORMALIZED = "normalized"    # v = normalize(v + Σr)
    MOMENTUM = "momentum"        # v = v + α*r + β*r_prev (momentum)


@dataclass
class CompositionConfig:
    """Configuração do raciocínio composicional."""
    
    # Modo de resíduo
    residual_mode: ResidualMode = ResidualMode.HEBBIAN
    composition_strategy: CompositionStrategy = CompositionStrategy.ADDITIVE
    
    # Escalas
    residual_scale: float = 0.1          # Escala global dos resíduos
    hebbian_scale: float = 0.15          # Escala específica Hebbiana
    attention_scale: float = 0.1         # Escala específica Attention
    field_scale: float = 0.05            # Escala específica Campo
    
    # Hebbian
    hebbian_top_k: int = 8               # Vizinhos Hebbianos a considerar
    hebbian_threshold: float = 0.01      # Peso mínimo para considerar
    
    # Attention
    attention_temperature: float = 1.0   # Temperatura do softmax
    attention_heads: int = 4             # Número de heads (se multi-head)
    
    # Campo
    field_epsilon: float = 1e-4          # Para gradiente numérico
    
    # Composição
    momentum_alpha: float = 0.9          # Peso do resíduo atual
    momentum_beta: float = 0.1           # Peso do resíduo anterior
    gate_bias: float = 0.5               # Bias do gate
    
    # Controle
    max_path_length: int = 50            # Máximo de nós no caminho
    early_stop_threshold: float = 0.001  # Para quando resíduo muito pequeno
    normalize_output: bool = True        # Normalizar vetor final
    
    # Interpretabilidade
    store_intermediate: bool = True      # Guardar estados intermediários
    decode_concepts: bool = True         # Tentar decodificar conceitos


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class ResidualContribution:
    """Contribuição de um único nó."""
    node_index: int
    coordinates: np.ndarray
    residual: np.ndarray
    weight: float                        # Peso/importância desta contribuição
    source: str                          # "hebbian", "attention", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionStep:
    """Um passo na composição."""
    step_index: int
    position: np.ndarray                 # Onde estamos
    velocity: np.ndarray                 # Para onde vamos
    residual: np.ndarray                 # Resíduo neste ponto
    cumulative: np.ndarray               # Vetor acumulado até aqui
    contributions: List[ResidualContribution]  # Detalhes das contribuições
    free_energy: float                   # F neste ponto
    concept_codes: Optional[np.ndarray] = None  # Códigos VQ-VAE


@dataclass
class CompositionalPath:
    """Resultado completo de uma travessia composicional."""
    
    # Entrada/Saída
    start_vector: np.ndarray
    target_vector: np.ndarray
    cumulative_vector: np.ndarray        # VETOR FINAL TRANSFORMADO
    
    # Caminho
    points: np.ndarray                   # [n_steps, dim] posições
    residuals: np.ndarray                # [n_steps, dim] resíduos
    
    # Métricas
    path_length: float                   # Comprimento geodésico
    total_residual_magnitude: float      # |Σr|
    n_steps: int
    converged: bool
    
    # Interpretabilidade
    composition_trace: List[str]         # Conceitos visitados (human-readable)
    steps: List[CompositionStep]         # Detalhes de cada passo
    
    # Análise
    dominant_contributions: List[Tuple[int, float]]  # (node_idx, weight)
    
    # Metadata
    config: CompositionConfig = None
    computation_time_ms: float = 0.0
    
    def summary(self) -> str:
        """Resumo human-readable do caminho."""
        lines = [
            f"=== Compositional Path ===",
            f"Steps: {self.n_steps}",
            f"Path length: {self.path_length:.4f}",
            f"Total residual: {self.total_residual_magnitude:.4f}",
            f"Converged: {self.converged}",
            f"",
            f"Trace: {' → '.join(self.composition_trace[:10])}",
        ]
        if len(self.composition_trace) > 10:
            lines.append(f"  ... and {len(self.composition_trace) - 10} more")
        
        return "\n".join(lines)


# =============================================================================
# REASONER PRINCIPAL
# =============================================================================

class CompositionalReasoner:
    """
    Raciocínio composicional via acumulação geodésica.
    
    O vetor é TRANSFORMADO durante a travessia, não apenas transportado.
    Cada nó visitado contribui semanticamente para o resultado final.
    """
    
    def __init__(
        self,
        bridge,                          # VQVAEManifoldBridge
        mycelial=None,                   # MycelialReasoning (opcional)
        geodesic_flow=None,              # BridgedGeodesicFlow (opcional)
        config: Optional[CompositionConfig] = None
    ):
        self.bridge = bridge
        self.mycelial = mycelial
        self.flow = geodesic_flow
        self.config = config or CompositionConfig()
        
        # Cache
        self._concept_cache = {}
        
        logger.info(
            f"CompositionalReasoner initialized: "
            f"mode={self.config.residual_mode.value}, "
            f"strategy={self.config.composition_strategy.value}"
        )
    
    # =========================================================================
    # API PRINCIPAL
    # =========================================================================
    
    def reason(
        self,
        start: np.ndarray,
        target: np.ndarray,
        mode: Optional[ResidualMode] = None
    ) -> CompositionalPath:
        """
        Executa raciocínio composicional de start até target.
        
        O vetor START é transformado ao longo do caminho até TARGET,
        acumulando contribuições semânticas de cada conceito visitado.
        
        Args:
            start: Vetor de partida (query)
            target: Vetor de destino (ou direção desejada)
            mode: Override do modo de resíduo
        
        Returns:
            CompositionalPath com vetor transformado e trace
        """
        import time
        t0 = time.time()
        
        mode = mode or self.config.residual_mode
        
        # Projetar para espaço do manifold se necessário
        if hasattr(self.bridge, '_project_to_latent'):
            start = self.bridge._project_to_latent(start)
            target = self.bridge._project_to_latent(target)
        
        # Computar caminho geodésico
        path_points, path_velocities = self._compute_path(start, target)
        
        # Acumular resíduos ao longo do caminho
        result = self._accumulate_residuals(
            start, target, path_points, path_velocities, mode
        )
        
        result.computation_time_ms = (time.time() - t0) * 1000
        result.config = self.config
        
        return result
    
    def reason_chain(
        self,
        start: np.ndarray,
        waypoints: List[np.ndarray],
        mode: Optional[ResidualMode] = None
    ) -> CompositionalPath:
        """
        Raciocínio em cadeia passando por waypoints específicos.
        
        Útil para forçar o raciocínio a passar por conceitos específicos.
        
        Args:
            start: Ponto inicial
            waypoints: Lista de pontos intermediários obrigatórios
            mode: Modo de resíduo
        
        Returns:
            CompositionalPath concatenado
        """
        # Projetar start para espaço do manifold
        if hasattr(self.bridge, '_project_to_latent'):
            start_proj = self.bridge._project_to_latent(start)
        else:
            start_proj = start
        
        all_points = []
        all_residuals = []
        all_steps = []
        all_trace = []
        
        current = start_proj.copy()
        total_length = 0.0
        
        targets = waypoints + [waypoints[-1]]  # Último é o destino final
        
        for i, target in enumerate(targets):
            # Raciocinar até próximo waypoint
            segment = self.reason(current, target, mode)
            
            all_points.extend(segment.points.tolist())
            all_residuals.extend(segment.residuals.tolist())
            all_steps.extend(segment.steps)
            all_trace.extend(segment.composition_trace)
            total_length += segment.path_length
            
            # Próximo segmento começa do vetor transformado
            current = segment.cumulative_vector
        
        return CompositionalPath(
            start_vector=start_proj,
            target_vector=targets[-1] if not hasattr(self.bridge, '_project_to_latent') 
                          else self.bridge._project_to_latent(targets[-1]),
            cumulative_vector=current,
            points=np.array(all_points),
            residuals=np.array(all_residuals),
            path_length=total_length,
            total_residual_magnitude=np.linalg.norm(current - start_proj),
            n_steps=len(all_points),
            converged=True,
            composition_trace=all_trace,
            steps=all_steps,
            dominant_contributions=self._find_dominant(all_steps)
        )
    
    def analogy(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> Tuple[np.ndarray, CompositionalPath]:
        """
        Analogia composicional: a:b :: c:?
        
        Encontra d tal que a relação a→b é análoga a c→d.
        Implementa "rei - homem + mulher = rainha" via geodésica.
        
        Args:
            a: Primeiro termo (ex: "rei")
            b: Segundo termo (ex: "homem")  
            c: Terceiro termo (ex: "mulher")
        
        Returns:
            (d, path): Quarto termo e caminho composicional
        """
        # Projetar para espaço do manifold
        if hasattr(self.bridge, '_project_to_latent'):
            a_proj = self.bridge._project_to_latent(a)
            b_proj = self.bridge._project_to_latent(b)
            c_proj = self.bridge._project_to_latent(c)
        else:
            a_proj, b_proj, c_proj = a, b, c
        
        # Computar transformação a → b
        path_ab = self.reason(a_proj, b_proj)
        transformation = path_ab.cumulative_vector - a_proj
        
        # Aplicar mesma transformação a c
        d = c_proj + transformation
        
        # Refinar via geodésica
        path_cd = self.reason(c_proj, d)
        
        return path_cd.cumulative_vector, path_cd
    
    # =========================================================================
    # COMPUTAÇÃO DE CAMINHO
    # =========================================================================
    
    def _compute_path(
        self,
        start: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa caminho geodésico de start a target.
        
        Se geodesic_flow disponível, usa ele.
        Senão, interpola linearmente (fallback).
        """
        if self.flow is not None:
            try:
                path = self.flow.shortest_path(start, target, n_iterations=5)
                return path.points, path.velocities
            except Exception as e:
                logger.warning(f"Geodesic flow failed: {e}, using linear")
        
        # Fallback: interpolação linear
        n_steps = min(
            self.config.max_path_length,
            max(10, int(np.linalg.norm(target - start) / 0.1))
        )
        
        points = np.zeros((n_steps, len(start)))
        velocities = np.zeros((n_steps, len(start)))
        
        for i in range(n_steps):
            t = i / (n_steps - 1)
            points[i] = start + t * (target - start)
            velocities[i] = (target - start) / n_steps
        
        return points, velocities
    
    # =========================================================================
    # ACUMULAÇÃO DE RESÍDUOS
    # =========================================================================
    
    def _accumulate_residuals(
        self,
        start: np.ndarray,
        target: np.ndarray,
        path_points: np.ndarray,
        path_velocities: np.ndarray,
        mode: ResidualMode
    ) -> CompositionalPath:
        """
        Núcleo do raciocínio composicional.
        
        Atravessa o caminho acumulando resíduos em cada ponto.
        """
        n_steps = len(path_points)
        dim = len(start)
        
        # Arrays de resultado
        residuals = np.zeros((n_steps, dim))
        steps = []
        trace = []
        
        # Estado
        v = start.copy()
        prev_residual = np.zeros(dim)
        total_length = 0.0
        
        for i in range(n_steps):
            point = path_points[i]
            velocity = path_velocities[i] if i < len(path_velocities) else np.zeros(dim)
            
            # Computar resíduo neste ponto
            residual, contributions = self._compute_residual(
                v, point, velocity, path_points, mode
            )
            
            # Aplicar estratégia de composição
            v, residual = self._apply_composition_strategy(
                v, residual, prev_residual
            )
            
            residuals[i] = residual
            prev_residual = residual
            
            # Comprimento do segmento
            if i > 0:
                total_length += np.linalg.norm(point - path_points[i-1])
            
            # Decodificar conceito (se configurado)
            concept_codes = None
            concept_name = f"step_{i}"
            if self.config.decode_concepts and self.bridge is not None:
                try:
                    codes, _ = self.bridge._find_codes_per_head(point)
                    concept_codes = codes
                    concept_name = self._decode_concept(codes)
                except:
                    pass
            
            trace.append(concept_name)
            
            # Energia livre aproximada
            F = self._approximate_free_energy(point)
            
            # Guardar passo
            if self.config.store_intermediate:
                steps.append(CompositionStep(
                    step_index=i,
                    position=point.copy(),
                    velocity=velocity.copy(),
                    residual=residual.copy(),
                    cumulative=v.copy(),
                    contributions=contributions,
                    free_energy=F,
                    concept_codes=concept_codes
                ))
            
            # Early stopping
            if np.linalg.norm(residual) < self.config.early_stop_threshold:
                if i > 5:  # Mínimo de passos
                    break
        
        # Normalizar se configurado
        if self.config.normalize_output:
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm
        
        return CompositionalPath(
            start_vector=start,
            target_vector=target,
            cumulative_vector=v,
            points=path_points[:len(steps)] if steps else path_points,
            residuals=residuals[:len(steps)] if steps else residuals,
            path_length=total_length,
            total_residual_magnitude=np.linalg.norm(v - start),
            n_steps=len(steps) if steps else n_steps,
            converged=True,
            composition_trace=trace[:len(steps)] if steps else trace,
            steps=steps,
            dominant_contributions=self._find_dominant(steps)
        )
    
    def _compute_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        velocity: np.ndarray,
        all_points: np.ndarray,
        mode: ResidualMode
    ) -> Tuple[np.ndarray, List[ResidualContribution]]:
        """
        Computa resíduo em um ponto específico.
        
        Retorna o vetor de resíduo e lista de contribuições para interpretabilidade.
        """
        contributions = []
        
        if mode == ResidualMode.HEBBIAN:
            residual = self._hebbian_residual(v, point, contributions)
        elif mode == ResidualMode.ATTENTION:
            residual = self._attention_residual(v, point, all_points, contributions)
        elif mode == ResidualMode.FIELD:
            residual = self._field_residual(v, point, contributions)
        elif mode == ResidualMode.PROJECTION:
            residual = self._projection_residual(v, point, velocity, contributions)
        elif mode == ResidualMode.HYBRID:
            residual = self._hybrid_residual(v, point, velocity, all_points, contributions)
        else:
            residual = np.zeros_like(v)
        
        return residual, contributions
    
    # =========================================================================
    # MODOS DE RESÍDUO
    # =========================================================================
    
    def _hebbian_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        contributions: List[ResidualContribution]
    ) -> np.ndarray:
        """
        Resíduo baseado em conexões Hebbianas.
        
        R = Σ w_ij · (neighbor_j - point)
        
        Vizinhos com conexão Hebbiana forte puxam o vetor.
        """
        residual = np.zeros_like(v)
        
        if self.mycelial is None or self.bridge is None:
            return residual
        
        try:
            # Encontrar códigos VQ-VAE do ponto
            codes, _ = self.bridge._find_codes_per_head(point)
            
            # Buscar conexões Hebbianas
            neighbors = self._get_hebbian_neighbors(codes)
            
            for neighbor_codes, weight in neighbors:
                if weight < self.config.hebbian_threshold:
                    continue
                
                # Reconstruir embedding do vizinho
                try:
                    neighbor_point = self.bridge.from_vqvae_codes(list(neighbor_codes))
                    direction = neighbor_point.coordinates - point
                    
                    contribution = weight * direction * self.config.hebbian_scale
                    residual += contribution
                    
                    contributions.append(ResidualContribution(
                        node_index=hash(neighbor_codes) % 10000,
                        coordinates=neighbor_point.coordinates,
                        residual=contribution,
                        weight=weight,
                        source="hebbian",
                        metadata={"codes": neighbor_codes}
                    ))
                except:
                    continue
            
        except Exception as e:
            logger.debug(f"Hebbian residual failed: {e}")
        
        return residual
    
    def _attention_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        all_points: np.ndarray,
        contributions: List[ResidualContribution]
    ) -> np.ndarray:
        """
        Resíduo via mecanismo de atenção.
        
        R = softmax(v · K^T / √d) · V
        
        O vetor atual "atende" a todos os pontos do caminho.
        """
        if len(all_points) < 2:
            return np.zeros_like(v)
        
        # Query = vetor atual
        # Keys = todos os pontos
        # Values = direções (pontos - ponto atual)
        
        # Scores
        scores = np.dot(all_points, v) / self.config.attention_temperature
        
        # Softmax estável
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Values = direções
        values = all_points - point
        
        # Weighted sum
        residual = np.sum(weights[:, None] * values, axis=0)
        residual = residual * self.config.attention_scale
        
        # Top contribuições
        top_k = min(5, len(weights))
        top_indices = np.argsort(weights)[-top_k:]
        
        for idx in top_indices:
            contributions.append(ResidualContribution(
                node_index=int(idx),
                coordinates=all_points[idx],
                residual=weights[idx] * values[idx] * self.config.attention_scale,
                weight=float(weights[idx]),
                source="attention"
            ))
        
        return residual
    
    def _field_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        contributions: List[ResidualContribution]
    ) -> np.ndarray:
        """
        Resíduo via gradiente do campo de energia livre.
        
        R = -∇F(x)
        
        O vetor é puxado na direção de menor energia.
        """
        residual = np.zeros_like(v)
        
        if self.bridge is None:
            return residual
        
        # Gradiente numérico
        epsilon = self.config.field_epsilon
        
        for i in range(min(len(point), 32)):  # Limitar dimensões para performance
            point_plus = point.copy()
            point_plus[i] += epsilon
            point_minus = point.copy()
            point_minus[i] -= epsilon
            
            F_plus = self._approximate_free_energy(point_plus)
            F_minus = self._approximate_free_energy(point_minus)
            
            residual[i] = -(F_plus - F_minus) / (2 * epsilon)
        
        residual = residual * self.config.field_scale
        
        contributions.append(ResidualContribution(
            node_index=-1,
            coordinates=point,
            residual=residual,
            weight=np.linalg.norm(residual),
            source="field",
            metadata={"free_energy": self._approximate_free_energy(point)}
        ))
        
        return residual
    
    def _projection_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        velocity: np.ndarray,
        contributions: List[ResidualContribution]
    ) -> np.ndarray:
        """
        Resíduo via projeção no espaço tangente.
        
        R = proj_{T_x M}(v - x)
        
        Projeta a diferença no espaço tangente local.
        """
        # Direção do movimento
        if np.linalg.norm(velocity) < 1e-8:
            return np.zeros_like(v)
        
        tangent = velocity / np.linalg.norm(velocity)
        
        # Diferença entre vetor atual e ponto
        diff = v - point
        
        # Projetar no tangente
        parallel = np.dot(diff, tangent) * tangent
        perpendicular = diff - parallel
        
        # Resíduo = componente perpendicular (corrige desvio)
        residual = -perpendicular * self.config.residual_scale
        
        contributions.append(ResidualContribution(
            node_index=-1,
            coordinates=point,
            residual=residual,
            weight=np.linalg.norm(perpendicular),
            source="projection"
        ))
        
        return residual
    
    def _hybrid_residual(
        self,
        v: np.ndarray,
        point: np.ndarray,
        velocity: np.ndarray,
        all_points: np.ndarray,
        contributions: List[ResidualContribution]
    ) -> np.ndarray:
        """
        Combinação ponderada de todos os modos.
        """
        heb_contrib = []
        att_contrib = []
        fld_contrib = []
        
        r_hebbian = self._hebbian_residual(v, point, heb_contrib)
        r_attention = self._attention_residual(v, point, all_points, att_contrib)
        r_field = self._field_residual(v, point, fld_contrib)
        
        # Pesos adaptativos baseados em magnitude
        mags = np.array([
            np.linalg.norm(r_hebbian),
            np.linalg.norm(r_attention),
            np.linalg.norm(r_field)
        ])
        
        if np.sum(mags) < 1e-8:
            return np.zeros_like(v)
        
        weights = mags / np.sum(mags)
        
        residual = (
            weights[0] * r_hebbian +
            weights[1] * r_attention +
            weights[2] * r_field
        )
        
        contributions.extend(heb_contrib)
        contributions.extend(att_contrib)
        contributions.extend(fld_contrib)
        
        return residual
    
    # =========================================================================
    # ESTRATÉGIAS DE COMPOSIÇÃO
    # =========================================================================
    
    def _apply_composition_strategy(
        self,
        v: np.ndarray,
        residual: np.ndarray,
        prev_residual: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica estratégia de composição ao vetor.
        
        Retorna (novo_vetor, residual_efetivo).
        """
        strategy = self.config.composition_strategy
        
        if strategy == CompositionStrategy.ADDITIVE:
            v_new = v + residual
            return v_new, residual
        
        elif strategy == CompositionStrategy.GATED:
            # Gate baseado na magnitude do vetor
            gate = 1.0 / (1.0 + np.exp(-np.linalg.norm(v) + self.config.gate_bias))
            effective_residual = gate * residual
            v_new = v + effective_residual
            return v_new, effective_residual
        
        elif strategy == CompositionStrategy.NORMALIZED:
            v_new = v + residual
            norm = np.linalg.norm(v_new)
            if norm > 1e-8:
                v_new = v_new / norm
            return v_new, residual
        
        elif strategy == CompositionStrategy.MOMENTUM:
            effective_residual = (
                self.config.momentum_alpha * residual +
                self.config.momentum_beta * prev_residual
            )
            v_new = v + effective_residual
            return v_new, effective_residual
        
        else:
            return v + residual, residual
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def _get_hebbian_neighbors(
        self,
        codes: np.ndarray
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Obtém vizinhos Hebbianos para um conjunto de códigos.
        """
        if self.mycelial is None:
            return []
        
        neighbors = []
        
        try:
            # Tentar API do MycelialReasoning
            if hasattr(self.mycelial, 'get_connected_codes'):
                return self.mycelial.get_connected_codes(
                    codes, 
                    top_k=self.config.hebbian_top_k
                )
            
            # Fallback: buscar no grafo diretamente
            if hasattr(self.mycelial, 'graph'):
                for h, c in enumerate(codes):
                    node = (h, int(c))
                    if node in self.mycelial.graph:
                        for neighbor, weight in self.mycelial.graph[node].items():
                            if weight >= self.config.hebbian_threshold:
                                # Reconstruir códigos completos
                                neighbor_codes = list(codes)
                                neighbor_codes[neighbor[0]] = neighbor[1]
                                neighbors.append((tuple(neighbor_codes), weight))
        except Exception as e:
            logger.debug(f"Failed to get Hebbian neighbors: {e}")
        
        # Ordenar por peso e limitar
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self.config.hebbian_top_k]
    
    def _approximate_free_energy(self, point: np.ndarray) -> float:
        """
        Aproxima energia livre como distância média aos atratores.
        """
        if self.bridge is None:
            return 0.0
        
        try:
            nearest = self.bridge.get_nearest_anchors(point, k=4)
            if nearest:
                return float(np.mean([dist for _, dist in nearest]))
        except:
            pass
        
        return 0.0
    
    def _decode_concept(self, codes: np.ndarray) -> str:
        """
        Tenta decodificar códigos em nome de conceito.
        
        Placeholder - em produção, usaria um mapeamento código→conceito.
        """
        code_str = ",".join(map(str, codes))
        
        # Cache lookup
        if code_str in self._concept_cache:
            return self._concept_cache[code_str]
        
        # Placeholder: retornar representação dos códigos
        concept = f"C[{code_str}]"
        self._concept_cache[code_str] = concept
        
        return concept
    
    def _find_dominant(
        self,
        steps: List[CompositionStep]
    ) -> List[Tuple[int, float]]:
        """
        Encontra contribuições dominantes ao longo do caminho.
        """
        if not steps:
            return []
        
        contribution_weights = {}
        
        for step in steps:
            for contrib in step.contributions:
                key = contrib.node_index
                if key not in contribution_weights:
                    contribution_weights[key] = 0.0
                contribution_weights[key] += contrib.weight
        
        # Ordenar por peso total
        sorted_contribs = sorted(
            contribution_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_contribs[:10]
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas do reasoner."""
        return {
            "config": {
                "residual_mode": self.config.residual_mode.value,
                "composition_strategy": self.config.composition_strategy.value,
                "residual_scale": self.config.residual_scale,
            },
            "has_bridge": self.bridge is not None,
            "has_mycelial": self.mycelial is not None,
            "has_flow": self.flow is not None,
            "concept_cache_size": len(self._concept_cache),
        }


# =============================================================================
# FACTORY E HELPERS
# =============================================================================

def create_reasoner(
    bridge,
    mycelial=None,
    flow=None,
    mode: str = "hebbian",
    strategy: str = "additive"
) -> CompositionalReasoner:
    """
    Factory function para criar reasoner configurado.
    
    Args:
        bridge: VQVAEManifoldBridge
        mycelial: MycelialReasoning (opcional)
        flow: BridgedGeodesicFlow (opcional)
        mode: "hebbian", "attention", "field", "projection", "hybrid"
        strategy: "additive", "gated", "normalized", "momentum"
    
    Returns:
        CompositionalReasoner configurado
    """
    config = CompositionConfig(
        residual_mode=ResidualMode(mode),
        composition_strategy=CompositionStrategy(strategy)
    )
    
    return CompositionalReasoner(bridge, mycelial, flow, config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Compositional Reasoning - Teste")
    print("=" * 60)
    
    # Importar dependências
    try:
        from vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig, ProjectionMode
    except ImportError:
        print("ERRO: vqvae_manifold_bridge.py não encontrado")
        exit(1)
    
    # Mock VQ-VAE
    class MockCodebook:
        def __init__(self, data):
            self._data = data
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self._data
    
    class MockQuantizer:
        def __init__(self):
            np.random.seed(42)
            self.codebooks = MockCodebook(
                np.random.randn(4, 256, 128).astype(np.float32)
            )
    
    class MockVQVAE:
        def __init__(self):
            self.quantizer = MockQuantizer()
        def get_codebook(self):
            return self.quantizer.codebooks.numpy()
    
    # Mock Mycelial (grafo Hebbiano simples)
    class MockMycelial:
        def __init__(self):
            self.graph = {}
            # Criar algumas conexões
            for h in range(4):
                for c in range(50):
                    node = (h, c)
                    self.graph[node] = {}
                    # Conectar a vizinhos
                    for dc in [-1, 1]:
                        neighbor = (h, (c + dc) % 256)
                        self.graph[node][neighbor] = np.random.random() * 0.5 + 0.5
    
    # Criar componentes
    print("\n1. Criando bridge...")
    bridge_config = BridgeConfig(
        projection_mode=ProjectionMode.WEIGHTED_ANCHORS,
        pull_strength=0.5
    )
    bridge = VQVAEManifoldBridge(bridge_config)
    bridge.connect_vqvae(MockVQVAE())
    print(f"   Anchors: {len(bridge.anchor_points)}")
    
    print("\n2. Criando mycelial mock...")
    mycelial = MockMycelial()
    print(f"   Graph nodes: {len(mycelial.graph)}")
    
    print("\n3. Criando reasoner...")
    config = CompositionConfig(
        residual_mode=ResidualMode.ATTENTION,  # Usar attention (não depende de mycelial real)
        composition_strategy=CompositionStrategy.ADDITIVE,
        residual_scale=0.1
    )
    reasoner = CompositionalReasoner(bridge, mycelial, None, config)
    
    # Teste: raciocínio simples
    print("\n4. Teste de raciocínio composicional...")
    start = np.random.randn(384).astype(np.float32)
    target = np.random.randn(384).astype(np.float32)
    
    result = reasoner.reason(start, target)
    
    print(f"\n{result.summary()}")
    print(f"\nVetor transformado:")
    print(f"   Shape: {result.cumulative_vector.shape}")
    print(f"   Mudança total: {np.linalg.norm(result.cumulative_vector - result.start_vector):.4f}")
    print(f"   Tempo: {result.computation_time_ms:.2f}ms")
    
    # Teste: analogia
    print("\n5. Teste de analogia (a:b :: c:?)...")
    a = np.random.randn(384).astype(np.float32)
    b = np.random.randn(384).astype(np.float32)
    c = np.random.randn(384).astype(np.float32)
    
    d, path = reasoner.analogy(a, b, c)
    print(f"   Resultado shape: {d.shape}")
    print(f"   Path length: {path.path_length:.4f}")
    
    # Teste: raciocínio em cadeia
    print("\n6. Teste de raciocínio em cadeia...")
    waypoints = [np.random.randn(384).astype(np.float32) for _ in range(3)]
    
    chain_result = reasoner.reason_chain(start, waypoints)
    print(f"   Total steps: {chain_result.n_steps}")
    print(f"   Total length: {chain_result.path_length:.4f}")
    print(f"   Concepts visited: {len(chain_result.composition_trace)}")
    
    # Stats
    print("\n7. Estatísticas:")
    stats = reasoner.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
