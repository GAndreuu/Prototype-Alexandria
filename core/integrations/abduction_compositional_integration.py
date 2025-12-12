"""
Abduction ↔ Compositional Integration
======================================

Conecta o Abduction Engine ao raciocínio composicional.

Hipóteses agora são representadas como caminhos geodésicos:
- Gap = descontinuidade no manifold
- Hipótese = geodésica proposta para conectar
- Validação = verificar se o caminho é "natural" (baixa energia)

Isso permite:
- Detectar gaps via curvatura anômala
- Gerar hipóteses como interpolações geodésicas
- Validar hipóteses pela energia do caminho
- Consolidar conhecimento deformando a métrica

Teoria:
    gap = região onde ∇F é alto mas não há atrator
    
    hypothesis = geodésica(source, target) que minimiza ∫F(γ)dt
    
    validation = F_path < threshold

Uso:
    from abduction_compositional_integration import AbductionCompositionalIntegration
    
    aci = AbductionCompositionalIntegration(bridge, abduction, compositional)
    
    # Detectar gaps geometricamente
    gaps = aci.detect_gaps_geometric(embeddings)
    
    # Gerar hipóteses como geodésicas
    hypotheses = aci.generate_geodesic_hypotheses(gap)

Autor: G (Alexandria Project)
Versão: 1.0
Fase: 2.1 - Raciocínio
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class AbductionCompositionalConfig:
    """Configuração da integração."""
    
    # Detecção de gaps
    gap_curvature_threshold: float = 0.3    # Curvatura mínima para gap
    gap_distance_threshold: float = 0.5      # Distância mínima entre clusters
    gap_energy_threshold: float = 0.7        # Energia alta = gap
    
    # Geração de hipóteses
    max_hypotheses_per_gap: int = 5
    hypothesis_path_samples: int = 10
    min_hypothesis_confidence: float = 0.3
    
    # Validação
    validation_energy_threshold: float = 0.5 # Caminho válido se F < threshold
    validation_curvature_bonus: float = 0.2  # Bônus por passar por atratores
    
    # Consolidação
    consolidation_deformation: float = 0.1   # Quanto deformar métrica ao consolidar


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class GeometricGap:
    """Gap de conhecimento detectado geometricamente."""
    gap_id: str
    gap_type: str                    # "curvature_anomaly", "energy_barrier", "disconnection"
    
    # Localização
    location: np.ndarray             # Centro do gap no manifold
    source_region: np.ndarray        # Região de origem
    target_region: np.ndarray        # Região de destino
    
    # Métricas geométricas
    curvature: float                 # Curvatura local
    energy_barrier: float            # Barreira de energia
    geodesic_distance: float         # Distância geodésica
    
    # Scores
    priority_score: float
    epistemic_value: float           # Valor de informação
    
    # Metadata
    affected_codes: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class GeodesicHypothesis:
    """Hipótese representada como caminho geodésico."""
    hypothesis_id: str
    hypothesis_text: str
    
    # Gap que resolve
    gap_id: str
    
    # Caminho
    source: np.ndarray
    target: np.ndarray
    geodesic_path: np.ndarray        # [n_steps, dim]
    path_residuals: np.ndarray       # Resíduos ao longo do caminho
    
    # Métricas
    path_length: float
    path_energy: float               # ∫F(γ)dt
    curvature_traversed: float
    attractors_visited: int
    
    # Scores
    confidence_score: float
    validation_score: float
    
    # Composição
    composition_trace: List[str] = field(default_factory=list)
    
    def is_valid(self, threshold: float = 0.5) -> bool:
        return self.path_energy < threshold and self.confidence_score > 0.3


# =============================================================================
# INTEGRAÇÃO PRINCIPAL
# =============================================================================

class AbductionCompositionalIntegration:
    """
    Integra Abduction Engine com Raciocínio Composicional.
    
    Transforma a detecção de gaps e geração de hipóteses em
    operações geométricas no manifold curvo.
    """
    
    def __init__(
        self,
        bridge,
        abduction_engine=None,
        compositional=None,
        config: Optional[AbductionCompositionalConfig] = None
    ):
        self.bridge = bridge
        self.abduction = abduction_engine
        self.compositional = compositional
        self.config = config or AbductionCompositionalConfig()
        
        # Cache
        self._gap_cache = {}
        self._hypothesis_cache = {}
        
        logger.info("AbductionCompositionalIntegration initialized")
    
    # =========================================================================
    # DETECÇÃO DE GAPS
    # =========================================================================
    
    def detect_gaps_geometric(
        self,
        embeddings: List[np.ndarray],
        cluster_labels: Optional[np.ndarray] = None
    ) -> List[GeometricGap]:
        """
        Detecta knowledge gaps via análise geométrica.
        
        Gaps são regiões onde:
        1. Curvatura é baixa (sem atratores)
        2. Energia é alta (região não mapeada)
        3. Clusters estão desconectados geodesicamente
        
        Args:
            embeddings: Lista de embeddings do corpus
            cluster_labels: Labels de cluster (opcional)
        
        Returns:
            Lista de GeometricGap ordenada por prioridade
        """
        gaps = []
        
        # Converter para array
        embeddings = np.array(embeddings)
        
        # 1. Detectar anomalias de curvatura
        curvature_gaps = self._detect_curvature_anomalies(embeddings)
        gaps.extend(curvature_gaps)
        
        # 2. Detectar barreiras de energia
        energy_gaps = self._detect_energy_barriers(embeddings)
        gaps.extend(energy_gaps)
        
        # 3. Detectar desconexões entre clusters
        if cluster_labels is not None:
            disconnection_gaps = self._detect_disconnections(
                embeddings, cluster_labels
            )
            gaps.extend(disconnection_gaps)
        
        # Ordenar por prioridade
        gaps.sort(key=lambda g: g.priority_score, reverse=True)
        
        # Cache
        for gap in gaps:
            self._gap_cache[gap.gap_id] = gap
        
        return gaps
    
    def _detect_curvature_anomalies(
        self,
        embeddings: np.ndarray
    ) -> List[GeometricGap]:
        """Detecta regiões de curvatura anômala."""
        gaps = []
        
        # Amostrar pontos entre embeddings
        n_samples = min(100, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
        
        for idx in sample_indices:
            emb = embeddings[idx]
            
            # Projetar para manifold
            if hasattr(self.bridge, '_project_to_latent'):
                point = self.bridge._project_to_latent(emb)
            else:
                point = emb
            
            # Calcular curvatura
            try:
                g = self.bridge.compute_metric_deformation(point)
                curvature = np.trace(g) - len(point)
            except:
                curvature = 0.0
            
            # Gap se curvatura muito baixa (região não estruturada)
            if curvature < self.config.gap_curvature_threshold:
                # Encontrar vizinhos para delimitar região
                try:
                    nearest = self.bridge.get_nearest_anchors(point, k=2)
                    if nearest and len(nearest) >= 2:
                        source = nearest[0][0].coordinates
                        target = nearest[1][0].coordinates
                    else:
                        continue
                except:
                    continue
                
                gap = GeometricGap(
                    gap_id=f"curv_gap_{idx}",
                    gap_type="curvature_anomaly",
                    location=point,
                    source_region=source,
                    target_region=target,
                    curvature=curvature,
                    energy_barrier=self._compute_energy(point),
                    geodesic_distance=np.linalg.norm(target - source),
                    priority_score=1.0 - curvature,  # Menor curvatura = maior prioridade
                    epistemic_value=0.8,
                    description=f"Low curvature region ({curvature:.3f})"
                )
                gaps.append(gap)
        
        return gaps[:10]  # Top 10
    
    def _detect_energy_barriers(
        self,
        embeddings: np.ndarray
    ) -> List[GeometricGap]:
        """Detecta barreiras de energia entre regiões."""
        gaps = []
        
        # Amostrar pares de embeddings
        n_pairs = min(50, len(embeddings) * (len(embeddings) - 1) // 2)
        
        for _ in range(n_pairs):
            i, j = np.random.choice(len(embeddings), 2, replace=False)
            
            emb_i = embeddings[i]
            emb_j = embeddings[j]
            
            # Ponto médio
            midpoint = (emb_i + emb_j) / 2
            
            if hasattr(self.bridge, '_project_to_latent'):
                midpoint = self.bridge._project_to_latent(midpoint)
                source = self.bridge._project_to_latent(emb_i)
                target = self.bridge._project_to_latent(emb_j)
            else:
                source = emb_i
                target = emb_j
            
            # Energia no ponto médio
            energy_mid = self._compute_energy(midpoint)
            energy_source = self._compute_energy(source)
            energy_target = self._compute_energy(target)
            
            # Barreira = energia do meio - média das pontas
            barrier = energy_mid - (energy_source + energy_target) / 2
            
            if barrier > self.config.gap_energy_threshold:
                gap = GeometricGap(
                    gap_id=f"energy_gap_{i}_{j}",
                    gap_type="energy_barrier",
                    location=midpoint,
                    source_region=source,
                    target_region=target,
                    curvature=0.0,
                    energy_barrier=barrier,
                    geodesic_distance=np.linalg.norm(target - source),
                    priority_score=barrier,
                    epistemic_value=0.7,
                    description=f"Energy barrier ({barrier:.3f})"
                )
                gaps.append(gap)
        
        return gaps[:10]
    
    def _detect_disconnections(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> List[GeometricGap]:
        """Detecta desconexões entre clusters."""
        gaps = []
        
        unique_labels = np.unique(cluster_labels)
        
        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i+1:]:
                # Centroides dos clusters
                mask_i = cluster_labels == label_i
                mask_j = cluster_labels == label_j
                
                centroid_i = np.mean(embeddings[mask_i], axis=0)
                centroid_j = np.mean(embeddings[mask_j], axis=0)
                
                # Projetar
                if hasattr(self.bridge, '_project_to_latent'):
                    source = self.bridge._project_to_latent(centroid_i)
                    target = self.bridge._project_to_latent(centroid_j)
                else:
                    source = centroid_i
                    target = centroid_j
                
                # Verificar se há caminho "fácil"
                path_energy = self._compute_path_energy(source, target)
                
                # Distância geodésica vs euclidiana
                euclidean_dist = np.linalg.norm(target - source)
                
                # Se energia do caminho é alta, há desconexão
                if path_energy > self.config.gap_distance_threshold:
                    gap = GeometricGap(
                        gap_id=f"disconn_gap_{label_i}_{label_j}",
                        gap_type="disconnection",
                        location=(source + target) / 2,
                        source_region=source,
                        target_region=target,
                        curvature=0.0,
                        energy_barrier=path_energy,
                        geodesic_distance=euclidean_dist * (1 + path_energy),
                        priority_score=path_energy * 0.8,
                        epistemic_value=0.9,  # Alto valor epistêmico
                        description=f"Disconnection between clusters {label_i} and {label_j}"
                    )
                    gaps.append(gap)
        
        return gaps
    
    # =========================================================================
    # GERAÇÃO DE HIPÓTESES
    # =========================================================================
    
    def generate_geodesic_hypotheses(
        self,
        gap: GeometricGap
    ) -> List[GeodesicHypothesis]:
        """
        Gera hipóteses como caminhos geodésicos que resolvem o gap.
        
        Args:
            gap: GeometricGap a resolver
        
        Returns:
            Lista de GeodesicHypothesis candidatas
        """
        hypotheses = []
        
        source = gap.source_region
        target = gap.target_region
        
        # Gerar variações do caminho
        for i in range(self.config.max_hypotheses_per_gap):
            # Perturbar ligeiramente source e target
            noise_scale = 0.1 * (i + 1)
            source_perturbed = source + np.random.randn(len(source)) * noise_scale
            target_perturbed = target + np.random.randn(len(target)) * noise_scale
            
            # Computar caminho
            if self.compositional is not None:
                try:
                    result = self.compositional.reason(source_perturbed, target_perturbed)
                    path = result.points
                    path_length = result.path_length
                    residuals = result.residuals
                    trace = result.composition_trace
                except:
                    path, path_length, residuals, trace = self._simple_path(
                        source_perturbed, target_perturbed
                    )
            else:
                path, path_length, residuals, trace = self._simple_path(
                    source_perturbed, target_perturbed
                )
            
            # Métricas do caminho
            path_energy = self._compute_path_energy_from_points(path)
            curvature = self._compute_path_curvature(path)
            attractors = self._count_attractors_visited(path)
            
            # Confidence baseado na qualidade do caminho
            confidence = self._compute_hypothesis_confidence(
                path_energy, curvature, attractors, gap
            )
            
            # Gerar texto da hipótese
            hypothesis_text = self._generate_hypothesis_text(gap, trace)
            
            hypothesis = GeodesicHypothesis(
                hypothesis_id=f"hyp_{gap.gap_id}_{i}",
                hypothesis_text=hypothesis_text,
                gap_id=gap.gap_id,
                source=source_perturbed,
                target=target_perturbed,
                geodesic_path=path,
                path_residuals=residuals,
                path_length=path_length,
                path_energy=path_energy,
                curvature_traversed=curvature,
                attractors_visited=attractors,
                confidence_score=confidence,
                validation_score=0.0,  # Será preenchido na validação
                composition_trace=trace
            )
            
            hypotheses.append(hypothesis)
        
        # Ordenar por confidence
        hypotheses.sort(key=lambda h: h.confidence_score, reverse=True)
        
        # Filtrar por confidence mínimo
        hypotheses = [
            h for h in hypotheses 
            if h.confidence_score >= self.config.min_hypothesis_confidence
        ]
        
        # Cache
        for h in hypotheses:
            self._hypothesis_cache[h.hypothesis_id] = h
        
        return hypotheses
    
    def _simple_path(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, List[str]]:
        """Caminho simples (interpolação linear)."""
        n_steps = self.config.hypothesis_path_samples
        path = np.zeros((n_steps, len(source)))
        
        for i in range(n_steps):
            t = i / (n_steps - 1)
            path[i] = source + t * (target - source)
        
        length = np.linalg.norm(target - source)
        residuals = np.zeros_like(path)
        trace = [f"step_{i}" for i in range(n_steps)]
        
        return path, length, residuals, trace
    
    def _compute_hypothesis_confidence(
        self,
        path_energy: float,
        curvature: float,
        attractors: int,
        gap: GeometricGap
    ) -> float:
        """Calcula confidence da hipótese."""
        # Menor energia = maior confidence
        energy_factor = 1.0 / (1.0 + path_energy)
        
        # Mais curvatura = passa por regiões estruturadas
        curvature_factor = min(1.0, curvature * self.config.validation_curvature_bonus)
        
        # Mais atratores = caminho mais "natural"
        attractor_factor = min(1.0, attractors * 0.1)
        
        # Combinar
        confidence = 0.4 * energy_factor + 0.3 * curvature_factor + 0.3 * attractor_factor
        
        return confidence
    
    def _generate_hypothesis_text(
        self,
        gap: GeometricGap,
        trace: List[str]
    ) -> str:
        """Gera texto descritivo da hipótese."""
        trace_str = " → ".join(trace[:5])
        if len(trace) > 5:
            trace_str += f" → ... ({len(trace)-5} more)"
        
        return f"Hypothesis for {gap.gap_type}: Path via [{trace_str}]"
    
    # =========================================================================
    # VALIDAÇÃO
    # =========================================================================
    
    def validate_hypothesis_geometric(
        self,
        hypothesis: GeodesicHypothesis
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Valida hipótese verificando propriedades geométricas do caminho.
        
        Critérios:
        1. Energia do caminho abaixo do threshold
        2. Caminho passa por atratores (é "natural")
        3. Curvatura não é anormalmente alta (sem descontinuidades)
        
        Returns:
            (is_valid, metrics)
        """
        metrics = {}
        
        path = hypothesis.geodesic_path
        
        # 1. Energia do caminho
        path_energy = hypothesis.path_energy
        metrics['path_energy'] = path_energy
        energy_valid = path_energy < self.config.validation_energy_threshold
        
        # 2. Atratores visitados
        attractors = hypothesis.attractors_visited
        metrics['attractors_visited'] = attractors
        attractor_valid = attractors > 0
        
        # 3. Continuidade do caminho (sem saltos grandes)
        if len(path) > 1:
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            max_step = np.max(steps)
            mean_step = np.mean(steps)
            metrics['max_step'] = max_step
            metrics['mean_step'] = mean_step
            continuity_valid = max_step < 3 * mean_step  # Sem saltos abruptos
        else:
            continuity_valid = True
        
        # Score de validação
        validation_score = (
            0.4 * (1.0 - min(1.0, path_energy)) +
            0.3 * min(1.0, attractors * 0.2) +
            0.3 * float(continuity_valid)
        )
        
        hypothesis.validation_score = validation_score
        metrics['validation_score'] = validation_score
        
        is_valid = energy_valid and (attractor_valid or continuity_valid)
        
        return is_valid, metrics
    
    # =========================================================================
    # CONSOLIDAÇÃO
    # =========================================================================
    
    def consolidate_hypothesis_geometric(
        self,
        hypothesis: GeodesicHypothesis
    ) -> Dict[str, Any]:
        """
        Consolida hipótese validada deformando a métrica.
        
        O caminho da hipótese se torna "mais fácil" de percorrer,
        efetivamente criando conexão no manifold.
        
        Returns:
            Métricas da consolidação
        """
        metrics = {}
        
        path = hypothesis.geodesic_path
        
        # Deformar métrica ao longo do caminho
        deformation_strength = self.config.consolidation_deformation
        
        # Para cada ponto do caminho, adicionar deformação
        for point in path[::max(1, len(path)//5)]:  # Amostrar 5 pontos
            try:
                # Deformar no bridge (se suportado)
                if hasattr(self.bridge, 'add_deformation'):
                    self.bridge.add_deformation(
                        point,
                        intensity=deformation_strength
                    )
                    metrics['deformations_added'] = metrics.get('deformations_added', 0) + 1
            except:
                pass
        
        # Atualizar abduction engine se disponível
        if self.abduction is not None:
            try:
                # Converter para formato do abduction engine
                hyp_dict = {
                    'id': hypothesis.hypothesis_id,
                    'hypothesis_text': hypothesis.hypothesis_text,
                    'source': hypothesis.source.tolist() if hasattr(hypothesis.source, 'tolist') else list(hypothesis.source),
                    'target': hypothesis.target.tolist() if hasattr(hypothesis.target, 'tolist') else list(hypothesis.target),
                    'confidence_score': hypothesis.confidence_score,
                    'validation_score': hypothesis.validation_score
                }
                
                if hasattr(self.abduction, 'consolidate_knowledge'):
                    self.abduction.consolidate_knowledge(hyp_dict)
                    metrics['abduction_updated'] = True
            except Exception as e:
                logger.warning(f"Failed to update abduction: {e}")
                metrics['abduction_updated'] = False
        
        metrics['hypothesis_id'] = hypothesis.hypothesis_id
        metrics['confidence'] = hypothesis.confidence_score
        metrics['validation'] = hypothesis.validation_score
        
        return metrics
    
    # =========================================================================
    # UTILIDADES
    # =========================================================================
    
    def _compute_energy(self, point: np.ndarray) -> float:
        """Energia livre no ponto."""
        if self.bridge is None:
            return 0.0
        
        try:
            nearest = self.bridge.get_nearest_anchors(point, k=4)
            if nearest:
                return float(np.mean([d for _, d in nearest]))
        except:
            pass
        
        return 0.0
    
    def _compute_path_energy(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Energia total do caminho."""
        # Interpolar
        n_samples = 10
        total_energy = 0.0
        
        for i in range(n_samples):
            t = i / (n_samples - 1)
            point = source + t * (target - source)
            total_energy += self._compute_energy(point)
        
        return total_energy / n_samples
    
    def _compute_path_energy_from_points(self, path: np.ndarray) -> float:
        """Energia média ao longo do caminho."""
        if len(path) == 0:
            return 0.0
        
        energies = [self._compute_energy(p) for p in path]
        return float(np.mean(energies))
    
    def _compute_path_curvature(self, path: np.ndarray) -> float:
        """Curvatura total ao longo do caminho."""
        if self.bridge is None or len(path) == 0:
            return 0.0
        
        total = 0.0
        for point in path[::max(1, len(path)//5)]:
            try:
                g = self.bridge.compute_metric_deformation(point)
                total += max(0, np.trace(g) - len(point))
            except:
                pass
        
        return total
    
    def _count_attractors_visited(self, path: np.ndarray) -> int:
        """Conta quantos atratores o caminho passa perto."""
        if self.bridge is None or len(path) == 0:
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
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas da integração."""
        return {
            "has_bridge": self.bridge is not None,
            "has_abduction": self.abduction is not None,
            "has_compositional": self.compositional is not None,
            "cached_gaps": len(self._gap_cache),
            "cached_hypotheses": len(self._hypothesis_cache),
            "config": {
                "gap_curvature_threshold": self.config.gap_curvature_threshold,
                "max_hypotheses_per_gap": self.config.max_hypotheses_per_gap,
                "validation_energy_threshold": self.config.validation_energy_threshold
            }
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_abduction_compositional(
    bridge,
    abduction_engine=None,
    compositional=None,
    **config_kwargs
) -> AbductionCompositionalIntegration:
    """Factory function."""
    config = AbductionCompositionalConfig(**config_kwargs)
    return AbductionCompositionalIntegration(
        bridge, abduction_engine, compositional, config
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Abduction ↔ Compositional Integration - Teste")
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
    print("\n2. Criando integração Abduction ↔ Compositional...")
    aci = AbductionCompositionalIntegration(bridge)
    
    # Gerar embeddings de teste
    print("\n3. Detectando gaps geométricos...")
    embeddings = [np.random.randn(384).astype(np.float32) for _ in range(50)]
    
    gaps = aci.detect_gaps_geometric(embeddings)
    
    print(f"   Gaps detectados: {len(gaps)}")
    for gap in gaps[:3]:
        print(f"   - {gap.gap_id}: {gap.gap_type} (priority={gap.priority_score:.3f})")
    
    # Gerar hipóteses
    if gaps:
        print("\n4. Gerando hipóteses geodésicas...")
        hypotheses = aci.generate_geodesic_hypotheses(gaps[0])
        
        print(f"   Hipóteses geradas: {len(hypotheses)}")
        for hyp in hypotheses[:3]:
            print(f"   - {hyp.hypothesis_id}: conf={hyp.confidence_score:.3f}, energy={hyp.path_energy:.3f}")
        
        # Validar hipótese
        if hypotheses:
            print("\n5. Validando hipótese...")
            is_valid, metrics = aci.validate_hypothesis_geometric(hypotheses[0])
            
            print(f"   Valid: {is_valid}")
            for k, v in metrics.items():
                print(f"   {k}: {v}")
            
            # Consolidar
            if is_valid:
                print("\n6. Consolidando hipótese...")
                consolidation = aci.consolidate_hypothesis_geometric(hypotheses[0])
                for k, v in consolidation.items():
                    print(f"   {k}: {v}")
    
    # Stats
    print("\n7. Estatísticas:")
    stats = aci.stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)
