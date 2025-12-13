"""
Prototype Alexandria - Abduction Engine
Automatic hypothesis generation for knowledge expansion

Autor: Prototype Alexandria Team
Data: 2025-11-22
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime
import math
import random
import torch

from .causal_reasoning import CausalEngine, CausalGraph
# from core.agents.action_agent import ActionAgent, ActionType  # DEPRECATED
try:
    from core.agents.action import ActionAgent, ActionType
except ImportError:
    # Fallback ou mock se necessário durante refatoração
    pass


@dataclass
class Hypothesis:
    """Representa uma hipótese gerada pelo sistema"""
    id: str
    source_cluster: str
    target_cluster: str
    hypothesis_text: str
    confidence_score: float
    evidence_strength: float
    test_requirements: List[str]
    validation_status: str  # "pending", "validated", "rejected", "needs_evidence"
    created_at: datetime
    validated_at: Optional[datetime] = None
    test_results: Dict[str, Any] = None


@dataclass
class KnowledgeGap:
    """Representa uma lacuna no conhecimento identificadas pelo sistema"""
    gap_id: str
    gap_type: str  # "orphaned_cluster", "broken_chain", "missing_connection", "contradiction"
    description: str
    affected_clusters: List[str]
    priority_score: float
    candidate_hypotheses: List[str]
    detected_at: datetime


@dataclass
class ValidationTest:
    """Teste para validar uma hipótese"""
    test_id: str
    hypothesis_id: str
    test_type: str  # "pattern_match", "sequence_validation", "semantic_coherence", "evidence_synthesis"
    test_query: str
    expected_outcome: str
    actual_outcome: str
    passed: bool
    confidence: float
    timestamp: datetime


class AbductionEngine:
    """
    Motor de Abdução da ASI - V9
    
    Capaz de:
    - Identificar lacunas no conhecimento causal
    - Gerar hipóteses automaticamente
    - Validar hipóteses usando dados do SFS
    - Expandir o grafo causal com novas conexões
    - Aprender continuamente das validações
    """
    
    def __init__(self, sfs_path: str = "data/sfs_index.jsonl", fast_mode: bool = False):
        self.sfs_path = sfs_path
        self.fast_mode = fast_mode  # Skip slow LanceDB queries for evidence
        
        # Criar componentes mock se não existirem
        try:
            from core.topology.topology_engine import TopologyEngine
            from core.memory.semantic_memory import SemanticFileSystem
            engine = TopologyEngine()
            memory = SemanticFileSystem(engine) if not fast_mode else None
            self.causal_engine = CausalEngine(engine, memory)
            self.topology = engine
        except ImportError:
            # Fallback para demonstração
            self.causal_engine = None
            self.topology = None
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.knowledge_gaps: Dict[str, KnowledgeGap] = {}
        self.validation_tests: Dict[str, ValidationTest] = {}
        self.generation_history: List[Dict] = []
        self.learning_feedback: Dict[str, float] = {}
        
        # Evidence cache - populated from graph weights
        self._evidence_cache: Dict[Tuple[str, str], float] = {}
        
        # Configurações do motor de abdução
        self.min_confidence_threshold = 0.3
        self.max_hypotheses_per_cycle = 10
        self.gap_detection_sensitivity = 0.7
        self.hypothesis_diversity_factor = 0.8
        
        # Patterns predefinidos para geração de hipóteses
        self.causal_patterns = {
            "technological_evolution": ["innovation", "application", "improvement", "integration"],
            "scientific_discovery": ["hypothesis", "experiment", "validation", "theory"],
            "educational_progression": ["foundation", "prerequisite", "advancement", "specialization"],
            "research_methodology": ["question", "method", "result", "conclusion"],
            "problem_solving": ["problem", "approach", "solution", "implementation"]
        }
        
        # Templates para geração de hipóteses
        self.hypothesis_templates = [
            "{source} enables {target}",
            "{source} is a prerequisite for {target}",
            "{source} provides the foundation for {target}",
            "{source} leads to advances in {target}",
            "{source} is essential for understanding {target}",
            "{source} influences the development of {target}",
            "{source} creates opportunities for {target}",
            "{source} contributes to {target}"
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load cluster labels for readable hypotheses
        self.cluster_labels: Dict[str, str] = {}
        self._load_cluster_labels()
        
        if fast_mode:
            self.logger.info("AbductionEngine running in FAST MODE (no LanceDB queries)")
    
    def _load_cluster_labels(self, labels_path: str = "data/cluster_labels.json"):
        """Load human-readable labels for clusters."""
        import os
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cluster_labels = data.get('labels', {})
                self.cluster_sample_papers = data.get('sample_papers', {})
                self.logger.info(f"Loaded {len(self.cluster_labels)} cluster labels")
            except Exception as e:
                self.logger.warning(f"Could not load cluster labels: {e}")
                self.cluster_labels = {}
                self.cluster_sample_papers = {}
        else:
            self.logger.warning("No cluster_labels.json found - run cluster_labeler.py first")
            self.cluster_labels = {}
            self.cluster_sample_papers = {}
    
    def get_readable_label(self, cluster_id) -> str:
        """Convert cluster ID to human-readable label."""
        cluster_key = str(cluster_id)
        if cluster_key in self.cluster_labels:
            return self.cluster_labels[cluster_key]
        return f"cluster_{cluster_id}"
    
    def get_sample_papers(self, cluster_id, limit: int = 3) -> List[str]:
        """Get sample paper titles for a cluster."""
        cluster_key = str(cluster_id)
        if cluster_key in self.cluster_sample_papers:
            return self.cluster_sample_papers[cluster_key][:limit]
        return []
        
    def detect_knowledge_gaps(self, min_orphaned_score: float = 0.3) -> List[KnowledgeGap]:
        """
        Identifica lacunas no conhecimento causal
        
        Args:
            min_orphaned_score: Score mínimo para considerar um cluster órfão
            
        Returns:
            Lista de lacunas identificadas
        """
        self.logger.info("🔍 Detectando lacunas no conhecimento causal...")
        
        # 1. Identificar clusters órfãos (sem conexões causais)
        orphaned_clusters = self._find_orphaned_clusters(min_orphaned_score)
        
        # 2. Identificar cadeias quebradas (conexões incompletas)
        broken_chains = self._find_broken_chains()
        
        # 3. Identificar conexões faltantes óbvias
        missing_connections = self._find_missing_connections()
        
        # 4. Identificar contradições (futuro: implementar validação lógica)
        contradictions = self._find_potential_contradictions()
        
        # Consolidar todas as lacunas
        all_gaps = orphaned_clusters + broken_chains + missing_connections + contradictions
        
        # Armazenar lacunas identificadas
        for gap in all_gaps:
            self.knowledge_gaps[gap.gap_id] = gap
            
        self.logger.info(f"✅ Detectadas {len(all_gaps)} lacunas de conhecimento")
        return all_gaps
    
    def _find_orphaned_clusters(self, min_score: float) -> List[KnowledgeGap]:
        """Identifica clusters sem conexões causais significativas"""
        # Tentar carregar grafo existente primeiro
        if not self.causal_engine.causal_graph:
            self.causal_engine.load_causal_graph()  # Tenta carregar do arquivo
        
        # Se ainda não existe, construir
        if not self.causal_engine.causal_graph:
            self.causal_engine.build_causal_graph()
            
        gaps = []
        orphaned_count = 0
        
        for cluster_id, node in self.causal_engine.causal_graph.items():
            # Calcular score de isolamento do cluster
            isolation_score = self._calculate_isolation_score(cluster_id)
            
            if isolation_score >= min_score:
                gap_id = f"orphaned_{cluster_id}_{orphaned_count}"
                gaps.append(KnowledgeGap(
                    gap_id=gap_id,
                    gap_type="orphaned_cluster",
                    description=f"Cluster '{cluster_id}' está isolado (score: {isolation_score:.3f})",
                    affected_clusters=[cluster_id],
                    priority_score=isolation_score,
                    candidate_hypotheses=[],
                    detected_at=datetime.now()
                ))
                orphaned_count += 1
                
        return gaps
    
    def _calculate_isolation_score(self, cluster_id: str) -> float:
        """Calcula o score de isolamento de um cluster"""
        if not self.causal_engine.causal_graph:
            return 1.0
            
        node = self.causal_engine.causal_graph.get(cluster_id)
        if not node:
            return 1.0
            
        # Score baseado no número e força das conexões
        # node é um dicionário com outgoing connections
        out_connections = len(node) if isinstance(node, dict) else 0
        in_connections = 0  # Simplificado para demonstração
        total_connections = in_connections + out_connections
        
        if total_connections == 0:
            return 1.0  # Completamente isolado
            
        # Normalizar score (0 = bem conectado, 1 = isolado)
        max_expected_connections = 10  # Assumimos que um cluster bem conectado tem ~10 conexões
        isolation_factor = max(0, 1 - (total_connections / max_expected_connections))
        
        return isolation_factor
    
    def _find_broken_chains(self) -> List[KnowledgeGap]:
        """Identifica cadeias de conhecimento incompletas"""
        gaps = []
        broken_count = 0
        
        # Procurar por padrões A -> B -> C onde A e C estão conectados mas B está faltando
        # Implementação simplificada - pode ser expandida
        
        return gaps
    
    def _find_missing_connections(self) -> List[KnowledgeGap]:
        """Identifica conexões óbvias que estão faltando"""
        gaps = []
        missing_count = 0
        
        if not self.causal_engine.causal_graph:
            self.causal_engine.load_causal_graph()
        if not self.causal_engine.causal_graph:
            self.causal_engine.build_causal_graph()
            
        # Encontrar clusters semanticamente similares sem conexão causal
        clusters = list(self.causal_engine.causal_graph.keys())
        
        for i, cluster_a in enumerate(clusters):
            for j, cluster_b in enumerate(clusters[i+1:], i+1):
                # Verificar se há similaridade semântica
                similarity = self._calculate_semantic_similarity(cluster_a, cluster_b)
                
                if similarity > 0.6:  # Similaridade alta
                    # Verificar se já existe conexão causal
                    if not self.causal_engine.causal_graph.has_edge(cluster_a, cluster_b):
                        gap_id = f"missing_{cluster_a}_{cluster_b}_{missing_count}"
                        gaps.append(KnowledgeGap(
                            gap_id=gap_id,
                            gap_type="missing_connection",
                            description=f"Clusters similares '{cluster_a}' e '{cluster_b}' sem conexão causal (similaridade: {similarity:.3f})",
                            affected_clusters=[cluster_a, cluster_b],
                            priority_score=similarity * 0.8,  # Priority based on similarity
                            candidate_hypotheses=[],
                            detected_at=datetime.now()
                        ))
                        missing_count += 1
                        
        return gaps
    
    def _calculate_semantic_similarity(self, cluster_a: str, cluster_b: str) -> float:
        """Calcula similaridade semântica entre dois clusters"""
        # Implementação simplificada - usar embeddings reais no futuro
        # Por agora, usar similaridade de string e padrões known
        
        # Similaridade de string
        a_words = set(str(cluster_a).lower().split())
        b_words = set(str(cluster_b).lower().split())
        
        if not a_words or not b_words:
            return 0.0
            
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Bonus para padrões known
        pattern_bonus = self._get_pattern_bonus(cluster_a, cluster_b)
        
        return min(1.0, jaccard_similarity + pattern_bonus)
    
    def _get_pattern_bonus(self, cluster_a: str, cluster_b: str) -> float:
        """Calcula bonus para padrões conhecidos"""
        a_lower = str(cluster_a).lower()
        b_lower = str(cluster_b).lower()
        
        # Padrões científicos comuns
        science_patterns = {
            "math": ["physics", "engineering", "economics"],
            "physics": ["engineering", "chemistry", "astronomy"],
            "biology": ["medicine", "chemistry", "genetics"],
            "computer_science": ["mathematics", "engineering", "algorithms"],
            "psychology": ["neuroscience", "sociology", "medicine"],
            "chemistry": ["biology", "physics", "engineering"]
        }
        
        for base_field, related_fields in science_patterns.items():
            if base_field in a_lower:
                for related in related_fields:
                    if related in b_lower:
                        return 0.3  # Bonus para padrões científicos known
                        
        return 0.0
    
    def _find_potential_contradictions(self) -> List[KnowledgeGap]:
        """Identifica contradições potenciais no conhecimento"""
        # Implementação futura - requer lógica de validação mais avançada
        return []
    
    def generate_hypotheses(self, max_hypotheses: int = 10) -> List[Hypothesis]:
        """
        Gera hipóteses automaticamente para preencher lacunas identificadas
        
        Args:
            max_hypotheses: Número máximo de hipóteses a gerar
            
        Returns:
            Lista de hipóteses geradas
        """
        self.logger.info(f"🧠 Gerando até {max_hypotheses} hipóteses automaticamente...")
        
        if not self.knowledge_gaps:
            self.detect_knowledge_gaps()
            
        generated_hypotheses = []
        
        # Ordenar lacunas por prioridade
        sorted_gaps = sorted(
            self.knowledge_gaps.values(),
            key=lambda x: x.priority_score,
            reverse=True
        )
        
        for gap in sorted_gaps[:max_hypotheses]:
            # Gerar hipóteses para cada lacuna
            gap_hypotheses = self._generate_hypotheses_for_gap(gap)
            generated_hypotheses.extend(gap_hypotheses)
            
            # Atualizar lacuna com hipóteses candidatas
            gap.candidate_hypotheses = [h.id for h in gap_hypotheses]
            
        # Filtrar e armazenar hipóteses
        validated_hypotheses = []
        for hypothesis in generated_hypotheses:
            if hypothesis.confidence_score >= self.min_confidence_threshold:
                self.hypotheses[hypothesis.id] = hypothesis
                validated_hypotheses.append(hypothesis)
                
        # Registrar geração no histórico
        self.generation_history.append({
            "timestamp": datetime.now(),
            "gaps_processed": len(sorted_gaps),
            "hypotheses_generated": len(validated_hypotheses),
            "average_confidence": np.mean([h.confidence_score for h in validated_hypotheses]) if validated_hypotheses else 0.0
        })
        
        self.logger.info(f"✅ Geradas {len(validated_hypotheses)} hipóteses com confiança >= {self.min_confidence_threshold}")
        return validated_hypotheses
    
    def _generate_hypotheses_for_gap(self, gap: KnowledgeGap) -> List[Hypothesis]:
        """Gera hipóteses específicas para uma lacuna"""
        hypotheses = []
        
        if gap.gap_type == "orphaned_cluster":
            hypotheses.extend(self._generate_for_orphaned_cluster(gap))
        elif gap.gap_type == "missing_connection":
            hypotheses.extend(self._generate_for_missing_connection(gap))
        elif gap.gap_type == "broken_chain":
            hypotheses.extend(self._generate_for_broken_chain(gap))
            
        return hypotheses
    
    def _generate_for_orphaned_cluster(self, gap: KnowledgeGap) -> List[Hypothesis]:
        """Gera hipóteses para cluster órfão"""
        cluster = gap.affected_clusters[0]
        hypotheses = []
        
        # Procurar clusters relacionados para conectar
        related_clusters = self._find_related_clusters(cluster)
        
        for related_cluster in related_clusters[:3]:  # Top 3 relacionamentos
            hypothesis_id = f"hyp_{cluster}_{related_cluster}_{len(self.hypotheses)}"
            
            # Calcular confiança baseada na similaridade e contexto
            confidence = self._calculate_hypothesis_confidence(cluster, related_cluster, "orphaned_connection")
            
            # Get human-readable labels
            source_label = self.get_readable_label(cluster)
            target_label = self.get_readable_label(related_cluster)
            
            hypothesis_text = random.choice(self.hypothesis_templates).format(
                source=source_label,
                target=target_label
            )
            
            hypothesis = Hypothesis(
                id=hypothesis_id,
                source_cluster=f"{cluster} ({source_label})",
                target_cluster=f"{related_cluster} ({target_label})",
                hypothesis_text=hypothesis_text,
                confidence_score=confidence,
                evidence_strength=self._calculate_evidence_strength(cluster, related_cluster),
                test_requirements=["semantic_similarity", "temporal_sequence", "context_coherence"],
                validation_status="pending",
                created_at=datetime.now()
            )
            
            hypotheses.append(hypothesis)
            
        return hypotheses
    
    def _generate_for_missing_connection(self, gap: KnowledgeGap) -> List[Hypothesis]:
        """Gera hipóteses para conexão faltante"""
        cluster_a, cluster_b = gap.affected_clusters
        hypotheses = []
        
        # Gerar hipóteses bidirecionais
        for direction in [(cluster_a, cluster_b), (cluster_b, cluster_a)]:
            source, target = direction
            hypothesis_id = f"hyp_missing_{source}_{target}_{len(self.hypotheses)}"
            
            confidence = self._calculate_hypothesis_confidence(source, target, "missing_connection")
            
            # Get human-readable labels
            source_label = self.get_readable_label(source)
            target_label = self.get_readable_label(target)
            
            hypothesis_text = f"{source_label} influences the development of {target_label}"
            
            hypothesis = Hypothesis(
                id=hypothesis_id,
                source_cluster=f"{source} ({source_label})",
                target_cluster=f"{target} ({target_label})",
                hypothesis_text=hypothesis_text,
                confidence_score=confidence,
                evidence_strength=self._calculate_evidence_strength(source, target),
                test_requirements=["co_occurrence_analysis", "sequential_patterns", "expert_validation"],
                validation_status="pending",
                created_at=datetime.now()
            )
            
            hypotheses.append(hypothesis)
            
        return hypotheses
    
    def _generate_for_broken_chain(self, gap: KnowledgeGap) -> List[Hypothesis]:
        """Gera hipóteses para cadeia quebrada"""
        # Implementação futura para cadeias de conhecimento
        return []
    
    def _find_related_clusters(self, target_cluster: str) -> List[str]:
        """Encontra clusters relacionados a um cluster alvo"""
        related = []
        
        if not self.causal_engine.causal_graph:
            self.causal_engine.load_causal_graph()
        if not self.causal_engine.causal_graph:
            self.causal_engine.build_causal_graph()
        
        # 1. Primeiro: vizinhos DIRETOS no grafo causal (conexões existentes)
        if target_cluster in self.causal_engine.causal_graph:
            neighbors = self.causal_engine.causal_graph[target_cluster]
            if isinstance(neighbors, dict):
                # Ordenar por peso da conexão
                sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
                for neighbor, weight in sorted_neighbors[:3]:
                    related.append((neighbor, 1.0))  # Prioridade máxima
        
        # 2. Depois: clusters similares por semântica
        for cluster_id in self.causal_engine.causal_graph.keys():
            if cluster_id != target_cluster and cluster_id not in [r[0] for r in related]:
                similarity = self._calculate_semantic_similarity(target_cluster, cluster_id)
                if similarity > 0.2:  # Threshold mais baixo
                    related.append((cluster_id, similarity))
                    
        # Ordenar por similaridade e retornar top 5
        related.sort(key=lambda x: x[1], reverse=True)
        return [cluster for cluster, _ in related[:5]]
    
    def _calculate_hypothesis_confidence(self, source: str, target: str, gap_type: str) -> float:
        """Calcula confiança de uma hipótese"""
        # Base confidence
        base_confidence = 0.5
        
        # Ajuste baseado na similaridade semântica
        similarity = self._calculate_semantic_similarity(source, target)
        similarity_bonus = similarity * 0.3
        
        # Ajuste baseado no gap type
        type_adjustments = {
            "orphaned_connection": 0.1,
            "missing_connection": 0.4,
            "broken_chain": 0.15
        }
        type_bonus = type_adjustments.get(gap_type, 0.0)
        
        # Ajuste baseado em padrões known
        pattern_bonus = self._get_pattern_bonus(source, target) * 0.2
        
        # Calcular confiança final
        confidence = base_confidence + similarity_bonus + type_bonus + pattern_bonus
        
        # Adicionar ruído controlado para diversidade
        noise = random.uniform(-0.1, 0.1) * self.hypothesis_diversity_factor
        confidence += noise
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
    
    def _calculate_evidence_strength(self, source: str, target: str) -> float:
        """Calcula força da evidência para uma hipótese"""
        
        # FAST MODE: Use graph weights and semantic similarity instead of LanceDB
        if self.fast_mode:
            return self._calculate_evidence_fast(source, target)
        
        # Regular mode with LanceDB queries
        evidence_score = 0.0
        
        try:
            # Verificar co-ocorrência nos documentos
            co_occurrence_score = self._check_co_occurrence(source, target)
            
            # Verificar sequências temporais
            sequence_score = self._check_temporal_sequence(source, target)
            
            # Combinar evidências
            evidence_score = (co_occurrence_score * 0.4 + sequence_score * 0.6)
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular força da evidência: {e}")
            evidence_score = 0.3  # Default conservative
            
        return evidence_score
    
    def _calculate_evidence_fast(self, source: str, target: str) -> float:
        """
        Fast evidence calculation using graph weights and semantic similarity.
        No LanceDB queries - uses cached data only.
        """
        # Check cache first
        cache_key = (source, target)
        if cache_key in self._evidence_cache:
            return self._evidence_cache[cache_key]
        
        evidence_score = 0.3  # Base score
        
        # 1. Check if there's a direct connection in causal graph
        if self.causal_engine and self.causal_engine.causal_graph:
            graph = self.causal_engine.causal_graph
            if source in graph and isinstance(graph[source], dict):
                if target in graph[source]:
                    # Direct connection - strong evidence
                    weight = graph[source][target]
                    evidence_score = max(evidence_score, 0.6 + weight * 0.4)
            
            # Check reverse connection
            if target in graph and isinstance(graph[target], dict):
                if source in graph[target]:
                    weight = graph[target][source]
                    evidence_score = max(evidence_score, 0.5 + weight * 0.3)
        
        # 2. Use semantic similarity as supplementary evidence
        similarity = self._calculate_semantic_similarity(source, target)
        similarity_evidence = 0.2 + similarity * 0.4
        
        # Combine: take max of graph evidence and similarity evidence
        final_score = max(evidence_score, similarity_evidence)
        
        # Cache result
        self._evidence_cache[cache_key] = final_score
        
        return min(1.0, final_score)
    
    def consolidate_knowledge(self, hypothesis: Dict) -> bool:
        """
        Consolida uma hipótese validada na memória neural (Self-Learning).
        
        Args:
            hypothesis: Dict contendo 'source', 'target', 'relation'
            
        Returns:
            Success bool
        """
        if hypothesis.get('validation_score', 0) < 0.7:
            return False
            
        source_txt = hypothesis['source']
        target_txt = hypothesis['target']
        relation = hypothesis.get('relation', 'related')
        
        print(f"🧠 Consolidando conhecimento: {source_txt} -> {target_txt} ({relation})")
        
        try:
            # 1. Obter embeddings (usando TopologyEngine ou SFS)
            # Precisamos do vetor para passar pelo VQ-VAE
            # O AbductionEngine tem acesso ao topology? Sim, self.topology
            
            vec_source = self.topology.encode([source_txt])[0]
            vec_target = self.topology.encode([target_txt])[0]
            
            # 2. Converter para tensor
            t_source = torch.tensor(vec_source).float()
            t_target = torch.tensor(vec_target).float()
            
            # 3. Obter índices VQ-VAE
            # Precisamos de uma instância do MycelialVQVAE.
            # Se não tivermos, instanciamos (carrega pesos salvos)
            from core.reasoning.mycelial_reasoning import MycelialVQVAE
            
            # Otimização: Manter instância em cache se possível
            if not hasattr(self, '_neural_wrapper'):
                self._neural_wrapper = MycelialVQVAE.load_default()
                
            indices_source = self._neural_wrapper.encode(t_source).cpu().numpy().flatten()
            indices_target = self._neural_wrapper.encode(t_target).cpu().numpy().flatten()
            
            # 4. Aprender a transição (Hebbian Learning)
            # Se for causal, aprendemos a sequência
            if relation == 'causes':
                self._neural_wrapper.mycelial.observe_sequence(indices_source, indices_target)
            else:
                # Se for apenas correlação, aprendemos a co-ocorrência (observamos ambos juntos?)
                # Ou observamos a transição bidirecional
                self._neural_wrapper.mycelial.observe_sequence(indices_source, indices_target)
                self._neural_wrapper.mycelial.observe_sequence(indices_target, indices_source)
                
            # 5. Salvar estado
            self._neural_wrapper.mycelial.save_state()
            return True
            
        except Exception as e:
            print(f"❌ Erro na consolidação neural: {e}")
            return False

    def _check_co_occurrence(self, source: str, target: str) -> float:
        """
        Verifica se source e target aparecem juntos nos documentos usando LanceDB.
        Realiza busca vetorial pelo 'source' e verifica presença do 'target' nos resultados.
        """
        if not self.causal_engine or not hasattr(self.causal_engine, 'memory'):
            return 0.5  # Fallback se memória não estiver disponível
            
        try:
            # Buscar documentos relevantes para o termo fonte
            # Aumentamos o limit para ter uma amostra estatística melhor
            results = self.causal_engine.memory.retrieve(source, limit=50)
            
            if not results:
                return 0.1
                
            # Contar quantos resultados também contêm o termo alvo
            co_occurrence_count = 0
            target_lower = target.lower()
            
            for res in results:
                content = res.get('content', '').lower()
                if target_lower in content:
                    co_occurrence_count += 1
            
            # Calcular score: proporção de documentos do source que contêm target
            # Normalizamos para evitar scores muito baixos em corpus grandes
            score = co_occurrence_count / len(results)
            
            # Boost: se score > 0, normalizar para 0.3-1.0
            if score > 0:
                score = 0.3 + (score * 0.7)
                
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Erro ao verificar co-ocorrência real: {e}")
            return 0.3

    def _check_temporal_sequence(self, source: str, target: str) -> float:
        """
        Verifica se source aparece antes de target temporalmente.
        Compara timestamps médios dos documentos recuperados.
        """
        if not self.causal_engine or not hasattr(self.causal_engine, 'memory'):
            return 0.5
            
        try:
            # Recuperar documentos para ambos os termos
            source_docs = self.causal_engine.memory.retrieve(source, limit=20)
            target_docs = self.causal_engine.memory.retrieve(target, limit=20)
            
            if not source_docs or not target_docs:
                return 0.5
                
            # Função auxiliar para extrair timestamp
            def get_timestamp(doc):
                # Tentar pegar metadata 'created_at', 'published_date', 'timestamp' ou fallback para 'timestamp' do DB
                meta = doc.get('metadata', {})
                ts_str = meta.get('created_at') or meta.get('published_date') or meta.get('timestamp') or doc.get('timestamp')
                
                if ts_str:
                    try:
                        # Tentar parsing simples (ISO format)
                        return datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
                    except:
                        pass
                return datetime.now() # Fallback se não tiver data

            # Calcular média dos timestamps
            source_dates = [get_timestamp(d).timestamp() for d in source_docs]
            target_dates = [get_timestamp(d).timestamp() for d in target_docs]
            
            avg_source = np.mean(source_dates)
            avg_target = np.mean(target_dates)
            
            # Se source é mais antigo que target (avg_source < avg_target), score alto
            # Se forem iguais (mesma indexação), score neutro (0.5)
            diff_seconds = avg_target - avg_source
            
            if abs(diff_seconds) < 86400: # Menos de 1 dia de diferença
                return 0.5
            elif diff_seconds > 0: # Target é mais novo (Source -> Target faz sentido temporalmente)
                return 0.8
            else: # Target é mais antigo (Source -> Target inverte causalidade temporal)
                return 0.2
                
        except Exception as e:
            self.logger.warning(f"Erro ao verificar sequência temporal real: {e}")
            return 0.5
    
    def validate_hypothesis(self, hypothesis_id: str) -> bool:
        """
        Valida uma hipótese usando dados do SFS
        
        Args:
            hypothesis_id: ID da hipótese para validar
            
        Returns:
            True se a hipótese for validada, False caso contrário
        """
        if hypothesis_id not in self.hypotheses:
            self.logger.error(f"Hipótese {hypothesis_id} não encontrada")
            return False
            
        hypothesis = self.hypotheses[hypothesis_id]
        self.logger.info(f"🔬 Validando hipótese: {hypothesis.hypothesis_text}")
        
        validation_results = []
        overall_confidence = 0.0
        
        # Executar testes necessários
        for test_type in hypothesis.test_requirements:
            test_result = self._run_validation_test(hypothesis_id, test_type)
            validation_results.append(test_result)
            overall_confidence += test_result.confidence
            
        # Calcular confiança média
        overall_confidence /= len(validation_results) if validation_results else 0
        
        # Determinar status da validação
        passed_tests = sum(1 for test in validation_results if test.passed)
        total_tests = len(validation_results)
        
        if passed_tests / total_tests >= 0.6:  # 60% dos testes devem passar
            hypothesis.validation_status = "validated"
            hypothesis.validated_at = datetime.now()
            hypothesis.confidence_score = max(hypothesis.confidence_score, overall_confidence)
            self.logger.info(f"✅ Hipótese {hypothesis_id} validada com confiança {overall_confidence:.3f}")
            
            # --- SELF-FEEDING CYCLE ---
            # Trigger Internal Learning to consolidate this new knowledge
            try:
                # Simular vetores para o conceito de origem e destino (na prática viriam do SFS)
                # Criamos vetores que "aproximam" os dois conceitos no espaço latente
                source_vec = np.random.normal(0, 0.1, 384) + (hash(hypothesis.source_cluster) % 100) / 100.0
                target_vec = np.random.normal(0, 0.1, 384) + (hash(hypothesis.target_cluster) % 100) / 100.0
                
                # O "vetor de aprendizado" é a combinação que representa a nova conexão
                learning_vectors = [source_vec.tolist(), target_vec.tolist()]
                
                action_agent = ActionAgent()
                action_agent.execute_action(
                    ActionType.INTERNAL_LEARNING, 
                    {"vectors": learning_vectors}
                )
                self.logger.info(f"🧠 Conhecimento consolidado neuralmente via V2Learner")
            except Exception as e:
                self.logger.error(f"Falha no ciclo de auto-alimentação: {e}")
            # --------------------------
            
            return True
        elif overall_confidence < 0.3:
            hypothesis.validation_status = "rejected"
            self.logger.info(f"❌ Hipótese {hypothesis_id} rejeitada (confiança muito baixa: {overall_confidence:.3f})")
            return False
        else:
            hypothesis.validation_status = "needs_evidence"
            self.logger.info(f"⚠️  Hipótese {hypothesis_id} precisa de mais evidências (confiança: {overall_confidence:.3f})")
            return False
    
    def _run_validation_test(self, hypothesis_id: str, test_type: str) -> ValidationTest:
        """Executa um teste de validação específico"""
        hypothesis = self.hypotheses[hypothesis_id]
        test_id = f"test_{hypothesis_id}_{test_type}"
        
        test_query = f"Validar se {hypothesis.hypothesis_text} é supported by evidence"
        
        # Simular execução do teste (implementação real usaria SFS)
        if test_type == "semantic_similarity":
            confidence = self._validate_semantic_coherence(hypothesis.source_cluster, hypothesis.target_cluster)
            passed = confidence > 0.6
            outcome = f"Similarity score: {confidence:.3f}"
        elif test_type == "co_occurrence_analysis":
            confidence = self._validate_co_occurrence(hypothesis.source_cluster, hypothesis.target_cluster)
            passed = confidence > 0.5
            outcome = f"Co-occurrence score: {confidence:.3f}"
        elif test_type == "sequential_patterns":
            confidence = self._validate_sequential_patterns(hypothesis.source_cluster, hypothesis.target_cluster)
            passed = confidence > 0.4
            outcome = f"Sequential pattern score: {confidence:.3f}"
        else:
            confidence = 0.5  # Default
            passed = confidence > 0.5
            outcome = "Default validation outcome"
            
        test_result = ValidationTest(
            test_id=test_id,
            hypothesis_id=hypothesis_id,
            test_type=test_type,
            test_query=test_query,
            expected_outcome="Supporting evidence found",
            actual_outcome=outcome,
            passed=passed,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.validation_tests[test_id] = test_result
        return test_result
    
    def _validate_semantic_coherence(self, source: str, target: str) -> float:
        """Valida coerência semântica entre source e target"""
        return self._calculate_semantic_similarity(source, target)
    
    def _validate_co_occurrence(self, source: str, target: str) -> float:
        """Valida se source e target co-ocorrem nos documentos"""
        return self._check_co_occurrence(source, target)
    
    def _validate_sequential_patterns(self, source: str, target: str) -> float:
        """Valida padrões sequenciais entre source e target"""
        return self._check_temporal_sequence(source, target)
    
    def expand_knowledge_graph(self, validated_hypotheses: List[str]) -> Dict[str, Any]:
        """
        Expande o grafo causal com hipóteses validadas
        
        Args:
            validated_hypotheses: Lista de IDs de hipóteses validadas
            
        Returns:
            Estatísticas da expansão
        """
        self.logger.info(f"📈 Expandindo grafo causal com {len(validated_hypotheses)} hipóteses validadas...")
        
        if not self.causal_engine.causal_graph:
            self.causal_engine.load_causal_graph()
        if not self.causal_engine.causal_graph:
            self.causal_engine.build_causal_graph()
            
        expansion_stats = {
            "new_connections": 0,
            "strengthened_connections": 0,
            "clusters_affected": set(),
            "confidence_improvements": {}
        }
        
        for hypothesis_id in validated_hypotheses:
            if hypothesis_id not in self.hypotheses:
                continue
                
            hypothesis = self.hypotheses[hypothesis_id]
            
            if hypothesis.validation_status == "validated":
                # Adicionar ou fortalecer conexão no grafo causal
                source = hypothesis.source_cluster
                target = hypothesis.target_cluster
                confidence = hypothesis.confidence_score
                
                if not self.causal_engine.causal_graph.has_edge(source, target):
                    # Nova conexão
                    self.causal_engine.causal_graph.add_edge(
                        source, 
                        target, 
                        confidence,
                        "abduction_generated"
                    )
                    expansion_stats["new_connections"] += 1
                else:
                    # Fortalecer conexão existente
                    existing_edge = self.causal_engine.causal_graph.edges[(source, target)]
                    new_confidence = (existing_edge.confidence + confidence) / 2
                    existing_edge.confidence = new_confidence
                    expansion_stats["strengthened_connections"] += 1
                    
                expansion_stats["clusters_affected"].add(source)
                expansion_stats["clusters_affected"].add(target)
                expansion_stats["confidence_improvements"][f"{source}->{target}"] = confidence
        
        # Converter set para list para JSON serialization
        expansion_stats["clusters_affected"] = list(expansion_stats["clusters_affected"])
        
        self.logger.info(f"✅ Expansão concluída: {expansion_stats['new_connections']} novas conexões, "
                        f"{expansion_stats['strengthened_connections']} conexões fortalecidas")
        
        return expansion_stats
    
    def run_abduction_cycle(self, max_hypotheses: int = 10) -> Dict[str, Any]:
        """
        Executa um ciclo completo de abdução
        
        Returns:
            Estatísticas completas do ciclo
        """
        self.logger.info("🚀 Iniciando ciclo completo de abdução...")
        
        cycle_start = datetime.now()
        
        # 1. Detectar lacunas
        gaps = self.detect_knowledge_gaps()
        
        # 2. Gerar hipóteses
        hypotheses = self.generate_hypotheses(max_hypotheses)
        
        # 3. Validar hipóteses
        validated_ids = []
        validation_stats = {"validated": 0, "rejected": 0, "needs_evidence": 0}
        
        for hypothesis in hypotheses:
            if self.validate_hypothesis(hypothesis.id):
                validated_ids.append(hypothesis.id)
                validation_stats["validated"] += 1
            else:
                hypothesis = self.hypotheses[hypothesis.id]
                validation_stats[hypothesis.validation_status] += 1
        
        # 4. Expandir grafo causal
        expansion_stats = self.expand_knowledge_graph(validated_ids)
        
        # 5. Calcular estatísticas do ciclo
        cycle_duration = datetime.now() - cycle_start
        
        cycle_stats = {
            "cycle_duration_seconds": cycle_duration.total_seconds(),
            "gaps_detected": len(gaps),
            "hypotheses_generated": len(hypotheses),
            "hypotheses_validated": len(validated_ids),
            "validation_stats": validation_stats,
            "expansion_stats": expansion_stats,
            "average_confidence": np.mean([h.confidence_score for h in hypotheses]) if hypotheses else 0.0,
            "knowledge_gaps_remaining": len(self.knowledge_gaps),
            "total_hypotheses": len(self.hypotheses)
        }
        
        self.logger.info(f"🏁 Ciclo de abdução concluído em {cycle_duration.total_seconds():.2f}s")
        self.logger.info(f"📊 Estatísticas: {len(gaps)} lacunas, {len(hypotheses)} hipóteses, "
                        f"{len(validated_ids)} validadas")
        
        return cycle_stats
    
    def get_abduction_report(self) -> Dict[str, Any]:
        """Gera relatório completo do estado do motor de abdução"""
        
        # Estatísticas gerais
        total_hypotheses = len(self.hypotheses)
        validated_hypotheses = sum(1 for h in self.hypotheses.values() if h.validation_status == "validated")
        rejected_hypotheses = sum(1 for h in self.hypotheses.values() if h.validation_status == "rejected")
        
        # Estatísticas de lacunas
        total_gaps = len(self.knowledge_gaps)
        gap_types = {}
        for gap in self.knowledge_gaps.values():
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1
            
        # Confiança média
        confidence_scores = [h.confidence_score for h in self.hypotheses.values()]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Evolução do sistema
        recent_cycles = self.generation_history[-5:] if len(self.generation_history) >= 5 else self.generation_history
        cycle_trends = {
            "hypotheses_per_cycle": [cycle["hypotheses_generated"] for cycle in recent_cycles],
            "average_confidence_trend": [cycle["average_confidence"] for cycle in recent_cycles]
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": {
                "total_hypotheses": total_hypotheses,
                "validated_hypotheses": validated_hypotheses,
                "rejected_hypotheses": rejected_hypotheses,
                "validation_rate": validated_hypotheses / total_hypotheses if total_hypotheses > 0 else 0,
                "rejection_rate": rejected_hypotheses / total_hypotheses if total_hypotheses > 0 else 0
            },
            "knowledge_gaps": {
                "total_gaps": total_gaps,
                "gap_types": gap_types,
                "average_priority": np.mean([gap.priority_score for gap in self.knowledge_gaps.values()]) if self.knowledge_gaps else 0
            },
            "confidence_metrics": {
                "average_confidence": avg_confidence,
                "high_confidence_hypotheses": sum(1 for score in confidence_scores if score > 0.7),
                "low_confidence_hypotheses": sum(1 for score in confidence_scores if score < 0.3)
            },
            "evolution_trends": cycle_trends,
            "recent_activity": self.generation_history[-10:] if self.generation_history else []
        }
        
        return report
    
    def save_state(self, filepath: str = "data/abduction_state.json"):
        """Salva o estado atual do motor de abdução"""
        state = {
            "hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()},
            "knowledge_gaps": {k: asdict(v) for k, v in self.knowledge_gaps.items()},
            "validation_tests": {k: asdict(v) for k, v in self.validation_tests.items()},
            "generation_history": [self._serialize_datetime(entry) for entry in self.generation_history],
            "learning_feedback": self.learning_feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"💾 Estado do motor de abdução salvo em {filepath}")
    
    def _serialize_datetime(self, obj):
        """Serializa objetos datetime para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        return obj


# Funções utilitárias para uso direto
def create_abduction_engine(sfs_path: str = "data/sfs_index.jsonl") -> AbductionEngine:
    """Cria uma instância do motor de abdução"""
    return AbductionEngine(sfs_path)


def run_quick_abduction_cycle(sfs_path: str = "data/sfs_index.jsonl", max_hypotheses: int = 5) -> Dict[str, Any]:
    """Executa um ciclo rápido de abdução"""
    engine = AbductionEngine(sfs_path)
    return engine.run_abduction_cycle(max_hypotheses)


if __name__ == "__main__":
    # Teste básico do motor de abdução
    print("🧠 Testando V9 Abduction Engine...")
    
    # Criar instância
    engine = AbductionEngine()
    
    # Executar ciclo de teste
    stats = engine.run_abduction_cycle(max_hypotheses=3)
    
    print(f"✅ Teste concluído! Estatísticas: {stats}")
    
    # Gerar relatório
    report = engine.get_abduction_report()
    print(f"📊 Relatório gerado com {len(report['recent_activity'])} ciclos de atividade")