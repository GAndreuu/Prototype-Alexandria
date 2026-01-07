"""
Unified Memory
==============

Abstração que combina memória semântica (core) com memória topológica (swarm).
Permite consultas unificadas e armazenamento coordenado de experiências.

Uso:
    from core.memory import UnifiedMemory
    
    memory = UnifiedMemory()
    result = memory.query("quantum entanglement")
"""

import logging
from typing import Dict, Optional, Any, Union, List, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Imports condicionais
try:
    from core.memory.semantic_memory import SemanticMemory
    SEMANTIC_AVAILABLE = True
except ImportError:
    logger.debug("SemanticMemory not available")
    SEMANTIC_AVAILABLE = False
    SemanticMemory = None

try:
    from swarm.core.memory import PersistentTopologicalMemory
    TOPOLOGICAL_AVAILABLE = True
except ImportError:
    logger.debug("PersistentTopologicalMemory not available")
    TOPOLOGICAL_AVAILABLE = False
    PersistentTopologicalMemory = None


# =============================================================================
# TIPOS
# =============================================================================

@dataclass
class SemanticResult:
    """Resultado de consulta à memória semântica."""
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class TrajectoryResult:
    """Resultado de consulta à memória de trajetórias."""
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    best_match_similarity: float = 0.0
    danger_zones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UnifiedQueryResult:
    """Resultado combinado de ambas as memórias."""
    semantic: Optional[SemanticResult] = None
    trajectories: Optional[TrajectoryResult] = None
    combined_confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    def has_results(self) -> bool:
        """Verifica se há resultados em alguma memória."""
        has_semantic = self.semantic and (self.semantic.concepts or self.semantic.relations)
        has_traj = self.trajectories and self.trajectories.trajectories
        return has_semantic or has_traj


@dataclass  
class UnifiedMemoryConfig:
    """Configuração da memória unificada."""
    semantic_weight: float = 0.5
    trajectory_weight: float = 0.5
    min_similarity: float = 0.6
    max_trajectories: int = 5
    semantic_memory_path: str = "data/semantic_memory.json"
    trajectory_memory_path: str = "data/swarm_memory.json"


# =============================================================================
# UNIFIED MEMORY
# =============================================================================

class UnifiedMemory:
    """
    Memória unificada: semântica + topológica.
    
    Combina:
    - SemanticMemory (core): conceitos, relações, conhecimento estruturado
    - PersistentTopologicalMemory (swarm): trajetórias de navegação, danger zones
    
    Benefícios:
    - Consulta única retorna ambos os tipos de informação
    - Experiências de navegação informam recuperação semântica
    - Armazenamento coordenado evita duplicação
    """
    
    def __init__(
        self,
        semantic_memory: Optional[Any] = None,
        topological_memory: Optional[Any] = None,
        topology_engine: Optional[Any] = None,
        config: Optional[UnifiedMemoryConfig] = None,
        field: Optional[Any] = None
    ):
        """
        Inicializa memória unificada.
        
        Args:
            semantic_memory: SemanticMemory instance (ou None para criar)
            topological_memory: PersistentTopologicalMemory instance (ou None para criar)
            topology_engine: TopologyEngine para encoding de conceitos
            config: Configuração
            field: PreStructuralField para sincronização com manifold
        """
        self.config = config or UnifiedMemoryConfig()
        self.topology = topology_engine
        self.field = field
        
        # Inicializar memória semântica
        if semantic_memory:
            self.semantic = semantic_memory
        elif SEMANTIC_AVAILABLE:
            try:
                self.semantic = SemanticMemory()
                logger.info("UnifiedMemory: SemanticMemory initialized")
            except Exception as e:
                logger.warning(f"Failed to init SemanticMemory: {e}")
                self.semantic = None
        else:
            self.semantic = None
        
        # Inicializar memória topológica
        if topological_memory:
            self.topological = topological_memory
        elif TOPOLOGICAL_AVAILABLE:
            try:
                self.topological = PersistentTopologicalMemory(
                    save_path=self.config.trajectory_memory_path
                )
                logger.info("UnifiedMemory: PersistentTopologicalMemory initialized")
            except Exception as e:
                logger.warning(f"Failed to init TopologicalMemory: {e}")
                self.topological = None
        else:
            self.topological = None
        
        # Stats
        self._stats = {
            "queries": 0,
            "semantic_hits": 0,
            "trajectory_hits": 0,
            "experiences_saved": 0
        }
    
    def query(
        self,
        concept: Union[str, np.ndarray],
        include_trajectories: bool = True,
        include_semantic: bool = True,
        top_k: int = 5
    ) -> UnifiedQueryResult:
        """
        Consulta ambas as memórias e combina resultados.
        
        Args:
            concept: Conceito (string) ou embedding (array)
            include_trajectories: Incluir trajetórias passadas
            include_semantic: Incluir conhecimento semântico
            top_k: Número máximo de resultados por memória
            
        Returns:
            UnifiedQueryResult com resultados combinados
        """
        self._stats["queries"] += 1
        
        result = UnifiedQueryResult()
        concept_emb = self._resolve_embedding(concept)
        
        # 1. Consultar memória semântica
        if include_semantic and self.semantic:
            try:
                result.semantic = self._query_semantic(concept, concept_emb, top_k)
                if result.semantic.concepts or result.semantic.relations:
                    result.sources.append("semantic")
                    self._stats["semantic_hits"] += 1
            except Exception as e:
                logger.debug(f"Semantic query failed: {e}")
        
        # 2. Consultar memória topológica
        if include_trajectories and self.topological:
            try:
                result.trajectories = self._query_topological(concept_emb, top_k)
                if result.trajectories.trajectories:
                    result.sources.append("topological")
                    self._stats["trajectory_hits"] += 1
            except Exception as e:
                logger.debug(f"Topological query failed: {e}")
        
        # 3. Calcular confiança combinada
        result.combined_confidence = self._compute_combined_confidence(result)
        
        return result
    
    def save_experience(
        self,
        trajectory: List[np.ndarray],
        start: Union[str, np.ndarray],
        target: Union[str, np.ndarray],
        success: bool,
        improvement: float,
        neurotypes_used: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Salva experiência de navegação em ambas as memórias.
        
        Args:
            trajectory: Lista de embeddings do caminho
            start: Conceito ou embedding inicial
            target: Conceito ou embedding alvo
            success: Se navegação foi bem-sucedida
            improvement: Melhoria de similaridade
            neurotypes_used: Contribuição de cada neurotipo
            metadata: Metadados adicionais
        """
        start_emb = self._resolve_embedding(start)
        target_emb = self._resolve_embedding(target)
        neurotypes = neurotypes_used or {}
        
        # 1. Salvar na memória topológica
        if self.topological:
            try:
                self.topological.save_trajectory(
                    trajectory=trajectory,
                    start=start_emb,
                    target=target_emb,
                    success=success,
                    efficiency=improvement,
                    mean_curvature=0.0,  # TODO: calcular se field disponível
                    neurotypes_used=neurotypes,
                    complexity=metadata.get('complexity', 0.5) if metadata else 0.5
                )
            except Exception as e:
                logger.warning(f"Failed to save trajectory: {e}")
        
        # 2. Se sucesso, extrair conceitos-chave para memória semântica
        if success and self.semantic and len(trajectory) > 2:
            try:
                self._extract_and_store_concepts(trajectory, start, target)
            except Exception as e:
                logger.debug(f"Failed to extract concepts: {e}")
                
        # 3. Manifold Update (Topology Sync)
        if self.field is not None and target_emb is not None:
             # Trigger topological update
             # Se intensity não vier dos metadados, assume base 1.0 ou condicional
             intensity = 1.0
             if metadata:
                 intensity = metadata.get('intensity', 1.0)
                 
             try:
                 # "Deforma" o manifold no ponto alvo gerando atrator
                 self.field.trigger(target_emb, intensity=intensity)
                 logger.debug(f"UnifiedMemory: Synced target to manifold (I={intensity})")
             except Exception as e:
                 logger.warning(f"UnifiedMemory: Failed to sync to manifold: {e}")
        
        self._stats["experiences_saved"] += 1
    
    def find_similar_experiences(
        self,
        start: Union[str, np.ndarray],
        target: Union[str, np.ndarray],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Encontra experiências passadas similares.
        
        Args:
            start: Ponto inicial
            target: Ponto alvo
            top_k: Número de experiências
            
        Returns:
            Lista de experiências similares com metadados
        """
        if not self.topological:
            return []
        
        start_emb = self._resolve_embedding(start)
        target_emb = self._resolve_embedding(target)
        
        try:
            return self.topological.find_similar_trajectories(
                start_emb=start_emb,
                target_emb=target_emb,
                top_k=top_k,
                min_similarity=self.config.min_similarity
            )
        except Exception as e:
            logger.debug(f"Failed to find similar experiences: {e}")
            return []
    
    def _query_semantic(
        self,
        concept: Union[str, np.ndarray],
        concept_emb: np.ndarray,
        top_k: int
    ) -> SemanticResult:
        """Consulta memória semântica."""
        result = SemanticResult()
        
        if not self.semantic:
            return result
        
        # Tentar diferentes métodos de consulta
        if hasattr(self.semantic, 'retrieve'):
            data = self.semantic.retrieve(concept if isinstance(concept, str) else concept_emb)
            if isinstance(data, dict):
                result.concepts = data.get('concepts', [])
                result.relations = data.get('relations', [])
                result.confidence = data.get('confidence', 0.5)
            elif isinstance(data, list):
                result.concepts = data[:top_k]
                result.confidence = 0.5
        
        elif hasattr(self.semantic, 'search'):
            results = self.semantic.search(concept_emb, top_k=top_k)
            result.concepts = results
            result.confidence = 0.5
        
        return result
    
    def _query_topological(
        self,
        concept_emb: np.ndarray,
        top_k: int
    ) -> TrajectoryResult:
        """Consulta memória topológica."""
        result = TrajectoryResult()
        
        if not self.topological:
            return result
        
        # Buscar trajetórias similares (usando mesmo ponto como start/target)
        trajectories = self.topological.find_similar_trajectories(
            start_emb=concept_emb,
            target_emb=concept_emb,
            top_k=top_k,
            min_similarity=self.config.min_similarity
        )
        
        if trajectories:
            result.trajectories = trajectories
            result.best_match_similarity = max(
                t.get('similarity', 0) for t in trajectories
            ) if trajectories else 0.0
        
        # Verificar danger zones
        if hasattr(self.topological, 'danger_zones'):
            result.danger_zones = list(self.topological.danger_zones.values())[:3]
        
        return result
    
    def _compute_combined_confidence(self, result: UnifiedQueryResult) -> float:
        """Calcula confiança combinada dos resultados."""
        confidences = []
        weights = []
        
        if result.semantic and result.semantic.confidence > 0:
            confidences.append(result.semantic.confidence)
            weights.append(self.config.semantic_weight)
        
        if result.trajectories and result.trajectories.best_match_similarity > 0:
            confidences.append(result.trajectories.best_match_similarity)
            weights.append(self.config.trajectory_weight)
        
        if not confidences:
            return 0.0
        
        # Média ponderada
        total_weight = sum(weights)
        return sum(c * w for c, w in zip(confidences, weights)) / total_weight
    
    def _extract_and_store_concepts(
        self,
        trajectory: List[np.ndarray],
        start: Union[str, np.ndarray],
        target: Union[str, np.ndarray]
    ):
        """Extrai conceitos-chave de uma trajetória e armazena."""
        if not self.semantic or not hasattr(self.semantic, 'store'):
            return
        
        # Extrair pontos de inflexão (mudanças de direção significativas)
        if len(trajectory) < 3:
            return
        
        key_points = [trajectory[0]]  # Início
        
        for i in range(1, len(trajectory) - 1):
            prev_dir = trajectory[i] - trajectory[i-1]
            next_dir = trajectory[i+1] - trajectory[i]
            
            # Normalizar
            prev_norm = np.linalg.norm(prev_dir)
            next_norm = np.linalg.norm(next_dir)
            
            if prev_norm > 1e-9 and next_norm > 1e-9:
                prev_dir = prev_dir / prev_norm
                next_dir = next_dir / next_norm
                
                # Mudança de direção significativa
                angle_cos = np.dot(prev_dir, next_dir)
                if angle_cos < 0.7:  # > ~45 graus
                    key_points.append(trajectory[i])
        
        key_points.append(trajectory[-1])  # Fim
        
        # Armazenar relação entre pontos-chave
        for i, point in enumerate(key_points[:-1]):
            next_point = key_points[i + 1]
            self.semantic.store({
                'type': 'trajectory_waypoint',
                'embedding': point,
                'next_embedding': next_point,
                'success': True
            })
    
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
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória."""
        stats = {**self._stats}
        
        if self.semantic and hasattr(self.semantic, 'stats'):
            stats['semantic'] = self.semantic.stats()
        
        if self.topological and hasattr(self.topological, 'get_stats'):
            stats['topological'] = self.topological.get_stats()
        
        return stats
    
    def save(self):
        """Persiste ambas as memórias."""
        if self.semantic and hasattr(self.semantic, 'save'):
            self.semantic.save()
        
        if self.topological and hasattr(self.topological, 'save_to_disk'):
            self.topological.save_to_disk()
    
    def __del__(self):
        """Salva ao destruir."""
        try:
            self.save()
        except Exception:
            pass


# =============================================================================
# FACTORY
# =============================================================================

def create_unified_memory(
    topology_engine=None,
    config: Optional[UnifiedMemoryConfig] = None
) -> UnifiedMemory:
    """
    Factory para criar UnifiedMemory.
    
    Args:
        topology_engine: TopologyEngine para encoding
        config: Configuração opcional
        
    Returns:
        UnifiedMemory inicializado
    """
    return UnifiedMemory(
        topology_engine=topology_engine,
        config=config
    )
