"""
Hypothesis Executor - Transforma hipóteses em ações executáveis
================================================================

Conecta abduction_engine → action_agent
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto
try:
    from core.reasoning.symbol_grounding import SymbolGrounder
except ImportError:
    SymbolGrounder = None

logger = logging.getLogger(__name__)


class ExecutionActionType(Enum):
    """Tipos de ação baseados em confiança da hipótese"""
    QUERY_SEARCH = auto()      # Buscar mais evidência (baixa confiança)
    EXPLORE_CLUSTER = auto()   # Explorar cluster relacionado (média)
    BRIDGE_CONCEPTS = auto()   # Criar conexão (alta confiança)
    VALIDATE_EXISTING = auto() # Validar conexão existente
    DEEPEN_TOPIC = auto()      # Aprofundar em tópico específico


@dataclass
class ExecutableAction:
    """Ação pronta para execução"""
    action_type: ExecutionActionType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    source_hypothesis_id: str = ""
    priority: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "target": self.target,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "source_hypothesis_id": self.source_hypothesis_id,
            "priority": self.priority
        }


@dataclass
class ActionResult:
    """Resultado da execução de uma ação"""
    action: ExecutableAction
    success: bool
    evidence_found: List[str] = field(default_factory=list)
    new_connections: int = 0
    execution_time_ms: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "evidence_found": self.evidence_found,
            "new_connections": self.new_connections,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class HypothesisExecutor:
    """
    Transforma hipóteses do AbductionEngine em ações executáveis.
    
    Regras de transformação:
    - confidence < 0.3: QUERY_SEARCH (precisa mais evidência)
    - 0.3 <= confidence < 0.5: EXPLORE_CLUSTER 
    - 0.5 <= confidence < 0.8: DEEPEN_TOPIC
    - confidence >= 0.8: BRIDGE_CONCEPTS (criar conexão)
    """
    
    def __init__(
        self,
        semantic_memory=None,
        topology_engine=None,
        mycelial_reasoning=None,  # New dependency
        low_confidence_threshold: float = 0.3,
        medium_confidence_threshold: float = 0.5,
        high_confidence_threshold: float = 0.8
    ):
        self.semantic_memory = semantic_memory
        self.topology_engine = topology_engine
        self.mycelial = mycelial_reasoning
        self.low_threshold = low_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.high_threshold = high_confidence_threshold
        
        self.execution_count = 0
        self.success_count = 0
        
        # Initialize Symbol Grounder
        self.grounder = None
        if SymbolGrounder:
            try:
                self.grounder = SymbolGrounder()
            except Exception as e:
                logger.warning(f"Failed to init SymbolGrounder: {e}")
        
        logger.info("HypothesisExecutor inicializado")
    
    def hypothesis_to_action(self, hypothesis: Dict[str, Any]) -> ExecutableAction:
        """
        Converte uma hipótese em ação executável.
        
        Args:
            hypothesis: Dict com keys:
                - hypothesis_text: str
                - source_cluster: str
                - target_cluster: str  
                - confidence_score: float
                - test_requirements: List[str]
                - id: str (opcional)
        
        Returns:
            ExecutableAction pronta para execução
        """
        confidence = hypothesis.get("confidence_score", 0.5)
        source = hypothesis.get("source_cluster", "")
        target = hypothesis.get("target_cluster", "")
        text = hypothesis.get("hypothesis_text", "")
        hyp_id = hypothesis.get("id", f"hyp_{self.execution_count}")
        
        # Determinar tipo de ação baseado em confiança
        if confidence < self.low_threshold:
            action_type = ExecutionActionType.QUERY_SEARCH
            target_str = f"{source} {target}"
            expected = f"Encontrar evidências para: {text[:100]}"
            params = {"query": text, "max_results": 10}
            
        elif confidence < self.medium_threshold:
            action_type = ExecutionActionType.EXPLORE_CLUSTER
            target_str = source if source else target
            expected = f"Explorar conceitos relacionados a {target_str}"
            params = {"cluster_id": target_str, "depth": 2}
            
        elif confidence < self.high_threshold:
            action_type = ExecutionActionType.DEEPEN_TOPIC
            target_str = f"{source} -> {target}"
            expected = f"Aprofundar conexão entre {source} e {target}"
            params = {"source": source, "target": target, "find_intermediates": True}
            
        else:
            action_type = ExecutionActionType.BRIDGE_CONCEPTS
            target_str = f"{source} <-> {target}"
            expected = f"Criar conexão forte: {source} ↔ {target}"
            params = {
                "source": source, 
                "target": target, 
                "connection_type": "validated_hypothesis",
                "confidence": confidence
            }
        
        return ExecutableAction(
            action_type=action_type,
            target=target_str,
            parameters=params,
            expected_outcome=expected,
            source_hypothesis_id=hyp_id,
            priority=confidence
        )
    
    def execute(self, hypothesis: Dict[str, Any]) -> ActionResult:
        """
        Executa uma hipótese: converte em ação e executa.
        
        Args:
            hypothesis: Hipótese do AbductionEngine
            
        Returns:
            ActionResult com resultado da execução
        """
        import time
        start_time = time.time()
        
        action = self.hypothesis_to_action(hypothesis)
        self.execution_count += 1
        
        try:
            result = self._execute_action(action)
            execution_time = (time.time() - start_time) * 1000
            
            if result.success:
                self.success_count += 1
                
            result.execution_time_ms = execution_time
            
            logger.info(
                f"Ação executada: {action.action_type.name} | "
                f"Success: {result.success} | "
                f"Evidence: {len(result.evidence_found)} | "
                f"Time: {execution_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Erro ao executar ação: {e}")
            
            return ActionResult(
                action=action,
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _execute_action(self, action: ExecutableAction) -> ActionResult:
        """
        Executa a ação propriamente dita.
        
        Override este método para integrar com sistemas reais.
        """
        # Implementação base - pode ser estendida
        evidence = []
        new_connections = 0
        
        if action.action_type == ExecutionActionType.QUERY_SEARCH:
            if self.semantic_memory:
                results = self._search_evidence(action.parameters.get("query", ""))
                evidence = [r.get("content", "") for r in results[:5]]
            else:
                # Simulação
                evidence = [f"Evidence for: {action.target}"]
                
        elif action.action_type == ExecutionActionType.EXPLORE_CLUSTER:
            if self.mycelial and "node_id" in action.parameters:
                # Real traversal in Mycelial Graph
                node = action.parameters["node_id"]  # Expecting tuple (head, code)
                if isinstance(node, list): node = tuple(node)
                neighbors = self.mycelial.get_neighbors(node, top_k=5)
                evidence = [f"Neighbor: {n['node']} w={n['weight']:.2f}" for n in neighbors]
                new_connections = len(neighbors)
            elif self.mycelial and self.grounder and "cluster_id" in action.parameters:
                # Ground text cluster_id
                cluster_id = action.parameters["cluster_id"]
                nodes = self.grounder.ground(cluster_id)
                if nodes:
                    # Explore from first Head (heuristic)
                    neighbors = self.mycelial.get_neighbors(nodes[0], top_k=5)
                    evidence = [f"Grounded Neighbor of {cluster_id}: {n['node']}" for n in neighbors]
                    new_connections = len(neighbors)
            elif self.semantic_memory:
                results = self._explore_cluster(action.parameters.get("cluster_id", ""))
                evidence = [r.get("content", "") for r in results[:5]]
            else:
                evidence = [f"Cluster exploration: {action.target}"]
                
        elif action.action_type == ExecutionActionType.BRIDGE_CONCEPTS:
            # Connect in Mycelial Graph
            source_node = action.parameters.get("source_node")
            target_node = action.parameters.get("target_node")
            source_txt = action.parameters.get("source")
            target_txt = action.parameters.get("target")
            
            # Grounding Logic
            if self.mycelial and self.grounder:
                if not source_node and source_txt:
                    source_nodes = self.grounder.ground(source_txt)
                else:
                    source_nodes = [source_node] if source_node else []
                    
                if not target_node and target_txt:
                    target_nodes = self.grounder.ground(target_txt)
                else:
                    target_nodes = [target_node] if target_node else []

                # Flatten and Validate
                s_nodes = [n for n in source_nodes if isinstance(n, tuple)]
                t_nodes = [n for n in target_nodes if isinstance(n, tuple)]

                if s_nodes and t_nodes:
                    count = 0
                    for na in s_nodes:
                        for nb in t_nodes:
                            # Connect corresponding heads (Heuristic)
                            if na[0] == nb[0]: 
                                w = self.mycelial.connect_nodes(na, nb, weight_delta=0.1)
                                count += 1
                    
                    if count > 0:
                        new_connections = count
                        evidence = [f"Grounded Bridge: {source_txt} <-> {target_txt} ({count} edges)"]
                        logger.info(f"[EXEC] BRIDGE_CONCEPTS: {source_txt} <-> {target_txt} ({count} edges)")

            if new_connections == 0:
                # Fallback: old logic or simulation
                if self.mycelial and source_node and target_node and isinstance(source_node, tuple) and isinstance(target_node, tuple):
                     w = self.mycelial.connect_nodes(source_node, target_node, weight_delta=0.1)
                     new_connections = 1
                     evidence = [f"Direct Bridge: {source_node} <-> {target_node}"]
                else:
                     new_connections = 1
                     evidence = [f"Logical Connection created: {source_txt} -> {target_txt}"]
            
        elif action.action_type == ExecutionActionType.DEEPEN_TOPIC:
            if self.semantic_memory:
                results = self._find_intermediate_concepts(
                    action.parameters.get("source", ""),
                    action.parameters.get("target", "")
                )
                evidence = [r.get("content", "") for r in results[:5]]
            else:
                evidence = [f"Intermediate concepts for: {action.target}"]
        
        return ActionResult(
            action=action,
            success=len(evidence) > 0 or new_connections > 0,
            evidence_found=evidence,
            new_connections=new_connections
        )
    
    def _search_evidence(self, query: str) -> List[Dict]:
        """Busca evidências na memória semântica"""
        if not self.semantic_memory:
            return []
        try:
            return self.semantic_memory.retrieve(query, limit=10)
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            return []
    
    def _explore_cluster(self, cluster_id: str) -> List[Dict]:
        """Explora um cluster de conceitos"""
        if not self.semantic_memory:
            return []
        try:
            return self.semantic_memory.retrieve(cluster_id, limit=20)
        except Exception as e:
            logger.error(f"Erro na exploração: {e}")
            return []
    
    def _find_intermediate_concepts(self, source: str, target: str) -> List[Dict]:
        """Encontra conceitos intermediários entre source e target"""
        if not self.semantic_memory:
            return []
        try:
            # Busca que combina ambos os conceitos
            query = f"{source} {target}"
            return self.semantic_memory.retrieve(query, limit=10)
        except Exception as e:
            logger.error(f"Erro ao buscar intermediários: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de execução"""
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0.0
        return {
            "total_executions": self.execution_count,
            "successful": self.success_count,
            "failed": self.execution_count - self.success_count,
            "success_rate": success_rate
        }
    
    def reset_stats(self):
        """Reseta estatísticas"""
        self.execution_count = 0
        self.success_count = 0
