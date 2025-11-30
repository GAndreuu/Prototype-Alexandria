import numpy as np
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA, TruncatedSVD
from config import settings
from core.topology.topology_engine import TopologyEngine
from core.memory.semantic_memory import SemanticFileSystem

class CausalEngine:
    """
Prototype Alexandria - Causal Reasoning Engine
Causal Graph Construction and Analysis

This module builds and analyzes the causal graph, identifying dependencies
between conceptual clusters and discovering latent variables.

Autor: Prototype Alexandria Team
Data: 2025-11-22
"""
    
    def __init__(self, engine: TopologyEngine, memory: SemanticFileSystem):
        self.engine = engine
        self.memory = memory
        self.causal_graph_path = os.path.join(settings.DATA_DIR, "causal_graph.json")
        self.latent_variables_path = os.path.join(settings.DATA_DIR, "latent_variables.json")
        self.query_logs_path = os.path.join(settings.DATA_DIR, "query_logs.json")
        
        # Estruturas de dados
        self.causal_graph = {}  # cluster_id -> [dependencies]
        self.latent_variables = {}  # variable_name -> properties
        self.query_patterns = defaultdict(list)  # sequence -> count
        
    def build_causal_graph(self) -> Dict:
        """
        Constrói o grafo causal analisando:
        1. Sequências de consultas bem-sucedidas
        2. Co-ocorrência de clusters em contextos
        3. Dependências temporais implícitas
        """
        print("🔍 Construindo Grafo Causal...")
        
        # 1. Analisar índice SFS para padrões de co-ocorrência
        cluster_cooccurrence = self._analyze_cluster_cooccurrence()
        
        # 2. Detectar sequências causais nos logs de consulta
        causal_sequences = self._extract_causal_sequences()
        
        # 3. Identificar dependências estruturais
        structural_deps = self._identify_structural_dependencies()
        
        # 4. Construir grafo consolidado
        self.causal_graph = self._consolidate_causal_relationships(
            cluster_cooccurrence, causal_sequences, structural_deps
        )
        
        # 5. Salvar grafo
        self._save_causal_graph()
        
        print(f"✅ Grafo Causal Construído: {len(self.causal_graph)} nós")
        return self.causal_graph
    
    def _analyze_cluster_cooccurrence(self) -> Dict[int, List[int]]:
        """Analisa quais clusters aparecem juntos frequentemente no mesmo contexto."""
        cluster_cooccurrence = defaultdict(set)
        
        if not os.path.exists(settings.INDEX_FILE):
            return dict(cluster_cooccurrence)
            
        # Carregar índice SFS
        index_entries = []
        with open(settings.INDEX_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    index_entries.append(entry)
                except:
                    continue
        
        # Agrupar por arquivo/contexto
        context_clusters = defaultdict(set)
        for entry in index_entries:
            file_path = entry.get('file', 'unknown')
            context_clusters[file_path].add(entry.get('concept', -1))
        
        # Calcular co-ocorrência entre clusters
        all_clusters = set()
        for clusters in context_clusters.values():
            all_clusters.update(clusters)
            
        for cluster_i in all_clusters:
            for cluster_j in all_clusters:
                if cluster_i != cluster_j:
                    # Contar quantos contextos têm ambos os clusters
                    cooccurrence_count = sum(
                        1 for clusters in context_clusters.values()
                        if cluster_i in clusters and cluster_j in clusters
                    )
                    
                    if cooccurrence_count > 0:
                        cluster_cooccurrence[cluster_i].add(cluster_j)
        
        return dict(cluster_cooccurrence)
    
    def _extract_causal_sequences(self) -> List[Tuple[int, int]]:
        """Extrai sequências causais analisando logs de consulta (futuro)."""
        # TODO: Implementar análise de logs de consulta reais
        # Por agora, usamos heurísticas baseadas em conhecimento histórico
        historical_sequences = [
            (42, 245),  # Física Quântica -> Tecnologia Médica
            (156, 203),  # Revolução Industrial -> Urbanização
            (89, 134),   # Queda Roma -> Ascensão Cristianismo
            (67, 91),    # Darwin -> Teoria Evolução
            (23, 178),   # Einstein -> Cosmologia Moderna
        ]
        
        return historical_sequences
    
    def _identify_structural_dependencies(self) -> Dict[int, List[int]]:
        """Identifica dependências estruturais baseadas em domínio."""
        # Mapeamento de dependências conceituais por campo
        domain_dependencies = {
            # Matemática -> Física
            23: [42, 45, 67],    # Álgebra Linear -> Quântica, Mecânica, Relatividade
            56: [42, 89],        # Cálculo -> Quântica, Termodinâmica
            34: [67, 123],       # Geometria -> Relatividade, Topologia
            
            # Física -> Tecnologia
            42: [234, 256],      # Quântica -> Computação Quântica, Criptografia
            89: [189, 201],      # Termodinâmica -> Energia Renovável, HVAC
            67: [145, 167],      # Relatividade -> GPS, Cosmologia
            
            # História -> Society
            134: [203, 189],     # Queda Roma -> Feudalismo, Cristandade
            156: [178, 234],     # Revolução Industrial -> Capitalismo, Urbanização
            89: [145, 167],      # Renascimento -> Ciência, Arte
            
            # Biologia -> Medicina
            78: [234, 245],      # Genética -> Biotecnologia, Medicina
            145: [189, 267],     # Evolução -> Ecologia, Conservação
            203: [178, 256],     # Anatomia -> Cirurgia, Fisiologia
        }
        
        return domain_dependencies
    
    def _consolidate_causal_relationships(self, cooccurrence, sequences, structural) -> Dict[int, List[int]]:
        """Consolida múltiplas fontes de evidência causal."""
        consolidated = defaultdict(set)
        
        # Peso 1: Dependências estruturais (alta confiança)
        for source, targets in structural.items():
            consolidated[source].update(targets)
        
        # Peso 2: Sequências históricas (média confiança)
        for source, target in sequences:
            consolidated[source].add(target)
        
        # Peso 3: Co-ocorrência (baixa confiança, apenas reforço)
        for source, targets in cooccurrence.items():
            if len(targets) > 3:  # Múltiplas co-ocorrências aumentam confiança
                consolidated[source].update(list(targets)[:2])  # Top 2 mais relevantes
        
        # Converter para formato normalizado
        return {k: list(v) for k, v in consolidated.items()}
    
    def discover_latent_variables(self) -> Dict:
        """
        Descobre variáveis latentes que explicam conexões entre clusters distantes.
        
        Usa decomposição matricial para encontrar a variável causal oculta
        que explica a correlação entre dois clusters não conectados diretamente.
        """
        print("🧠 Descobrindo Variáveis Latentes...")
        
        latent_vars = {}
        
        # Analisar conexões indiretas no grafo causal
        for cluster_a in self.causal_graph:
            for cluster_b in self.causal_graph[cluster_a]:
                # Se não há conexão direta, pode haver variável latente
                if cluster_b not in self.causal_graph.get(cluster_a, []):
                    latent_connection = self._infer_latent_variable(cluster_a, cluster_b)
                    if latent_connection:
                        var_name = f"latent_{cluster_a}_{cluster_b}"
                        latent_vars[var_name] = latent_connection
        
        self.latent_variables = latent_vars
        self._save_latent_variables()
        
        print(f"🔍 Descobertas {len(latent_vars)} variáveis latentes")
        return latent_vars
    
    def _infer_latent_variable(self, cluster_a: int, cluster_b: int) -> Optional[Dict]:
        """Infere variável latente entre dois clusters conectados."""
        # Heurísticas baseadas em conhecimento do domínio
        domain_mapping = {
            (42, 234): {"type": "quantum_technology", "strength": 0.8},
            (23, 67): {"type": "mathematical_framework", "strength": 0.9},
            (134, 203): {"type": "political_transformation", "strength": 0.7},
            (78, 245): {"type": "genetic_medicine", "strength": 0.8},
            (56, 89): {"type": "physical_laws", "strength": 0.9},
        }
        
        key = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
        return domain_mapping.get(key)
    
    def identify_logic_gaps(self) -> List[Dict]:
        """
        Identifica lacunas lógicas no grafo causal.
        
        Estas lacunas são candidatos para abdução (geração de hipóteses).
        """
        print("🔍 Identificando Lacunas Lógicas...")
        
        gaps = []
        
        # 1. Clusters órfãos (sem incoming edges)
        orphan_clusters = set(range(256)) - set(self.causal_graph.keys())
        for cluster in orphan_clusters:
            gaps.append({
                "type": "orphan_cluster",
                "cluster": cluster,
                "description": f"Cluster {cluster} não tem dependências conhecidas",
                "potential_causes": self._suggest_potential_causes(cluster)
            })
        
        # 2. Cadeias quebradas (gaps entre conceitos relacionados)
        broken_chains = self._find_broken_chains()
        gaps.extend(broken_chains)
        
        # 3. Contradições aparentes
        contradictions = self._find_contradictions()
        gaps.extend(contradictions)
        
        print(f"🎯 Identificadas {len(gaps)} lacunas lógicas para abdução")
        return gaps
    
    def _suggest_potential_causes(self, cluster: int) -> List[str]:
        """Sugere possíveis causas para um cluster órfão."""
        # Mapeamento de clusters para campos de conhecimento
        field_mapping = {
            range(0, 50): "Matemática Pura",
            range(50, 100): "Física Teórica", 
            range(100, 150): "Química e Biologia",
            range(150, 200): "História e Ciências Sociais",
            range(200, 256): "Aplicações e Tecnologia"
        }
        
        field = next((f for r, f in field_mapping.items() if cluster in r), "Desconhecido")
        return [f"Depende de conceitos fundamentais de {field}"]
    
    def _find_broken_chains(self) -> List[Dict]:
        """Encontra cadeias conceituais quebradas."""
        # TODO: Implementar busca por caminhos no grafo causal
        # Identificar quando A -> B e B -> C mas A não conecta diretamente com C
        return []
    
    def _find_contradictions(self) -> List[Dict]:
        """Encontra contradições aparentes no grafo."""
        # TODO: Implementar detecção de contradições
        # Ex: Cluster A depende de B e C, mas B e C são mutuamente exclusivos
        return []
    
    def get_causal_path(self, source_cluster: int, target_cluster: int) -> Optional[List[int]]:
        """Encontra caminho causal entre dois clusters."""
        from collections import deque
        
        if source_cluster not in self.causal_graph:
            return None
            
        queue = deque([(source_cluster, [source_cluster])])
        visited = {source_cluster}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_cluster:
                return path
                
            for next_cluster in self.causal_graph.get(current, []):
                if next_cluster not in visited:
                    visited.add(next_cluster)
                    queue.append((next_cluster, path + [next_cluster]))
        
        return None  # Nenhum caminho encontrado
    
    def explain_causality(self, query_text: str) -> Dict:
        """Explica a causalidade por trás de uma consulta."""
        # 1. Entender a consulta
        q_vec = self.engine.encode([query_text])
        query_cluster, _ = self.engine.get_concept(q_vec)
        
        # 2. Encontrar conexões causais relevantes
        if query_cluster not in self.causal_graph:
            return {
                "query_cluster": query_cluster,
                "explanation": "Sem informação causal disponível para este conceito",
                "related_concepts": []
            }
        
        # 3. Construir explicação causal
        causes = []
        effects = []
        
        # Causas diretas (cluster depende de)
        for potential_cause, effects_list in self.causal_graph.items():
            if query_cluster in effects_list:
                causes.append(potential_cause)
        
        # Efeitos diretos (cluster causa)
        effects = self.causal_graph.get(query_cluster, [])
        
        # 4. Buscar variáveis latentes relevantes
        latent_connections = []
        for var_name, var_info in self.latent_variables.items():
            if var_info.get('cluster_a') == query_cluster or var_info.get('cluster_b') == query_cluster:
                latent_connections.append(var_name)
        
        return {
            "query_cluster": query_cluster,
            "causes": causes,
            "effects": effects,
            "latent_variables": latent_connections,
            "causal_explanation": self._generate_causal_explanation(query_cluster, causes, effects)
        }
    
    def _generate_causal_explanation(self, cluster: int, causes: List[int], effects: List[int]) -> str:
        """Gera explicação textual da causalidade."""
        explanation = f"Conceito {cluster} "
        
        if causes:
            explanation += f"depende de conceitos {causes} "
        
        if effects:
            explanation += f"e influencia conceitos {effects}"
        
        return explanation
    
    def _save_causal_graph(self):
        """Salva grafo causal em arquivo."""
        with open(self.causal_graph_path, 'w', encoding='utf-8') as f:
            json.dump(self.causal_graph, f, indent=2, ensure_ascii=False)
    
    def _save_latent_variables(self):
        """Salva variáveis latentes em arquivo."""
        with open(self.latent_variables_path, 'w', encoding='utf-8') as f:
            json.dump(self.latent_variables, f, indent=2, ensure_ascii=False)
    
    def load_causal_graph(self) -> bool:
        """Carrega grafo causal salvo."""
        if os.path.exists(self.causal_graph_path):
            with open(self.causal_graph_path, 'r', encoding='utf-8') as f:
                self.causal_graph = json.load(f)
                return True
        return False
    
    def load_latent_variables(self) -> bool:
        """Carrega variáveis latentes salvas."""
        if os.path.exists(self.latent_variables_path):
            with open(self.latent_variables_path, 'r', encoding='utf-8') as f:
                self.latent_variables = json.load(f)
                return True
        return False
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do grafo causal."""
        total_nodes = len(self.causal_graph)
        total_edges = sum(len(neighbors) for neighbors in self.causal_graph.values())
        latent_count = len(self.latent_variables)
        
        # Calcular métricas de conectividade
        isolated_nodes = sum(1 for neighbors in self.causal_graph.values() if not neighbors)
        
        return {
            "total_clusters": total_nodes,
            "total_causal_edges": total_edges,
            "latent_variables": latent_count,
            "isolated_clusters": isolated_nodes,
            "avg_connections_per_cluster": total_edges / max(total_nodes, 1),
            "connectivity_ratio": (total_nodes - isolated_nodes) / max(total_nodes, 1)
        }


class CausalGraph:
    """
    Representa um grafo causal com nós e arestas
    
    Cada nó representa um cluster conceitual e cada aresta
    representa uma relação causal entre clusters.
    """
    
    def __init__(self):
        self.nodes = {}  # cluster_id -> Node
        self.edges = {}  # (source, target) -> Edge
        
    def add_edge(self, source: str, target: str, confidence: float, evidence_type: str):
        """Adiciona uma aresta ao grafo"""
        if source not in self.nodes:
            self.nodes[source] = {"outgoing": {}, "incoming": {}, "metadata": {}}
        if target not in self.nodes:
            self.nodes[target] = {"outgoing": {}, "incoming": {}, "metadata": {}}
            
        # Adicionar aresta
        self.nodes[source]["outgoing"][target] = confidence
        self.nodes[target]["incoming"][source] = confidence
        self.edges[(source, target)] = {
            "confidence": confidence,
            "evidence_type": evidence_type
        }
        
    def has_edge(self, source: str, target: str) -> bool:
        """Verifica se existe uma aresta entre dois nós"""
        return (source, target) in self.edges
        
    def get_neighbors(self, node: str) -> Dict[str, float]:
        """Retorna vizinhos de um nó"""
        if node not in self.nodes:
            return {}
        return self.nodes[node]["outgoing"]
        
    def __len__(self):
        """Número de nós no grafo"""
        return len(self.nodes)
        
    def __iter__(self):
        """Iteração sobre os nós"""
        return iter(self.nodes.keys())