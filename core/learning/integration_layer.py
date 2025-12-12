"""
Alexandria Integration Layer
=============================

Camada de integração que conecta todos os módulos cognitivos:
- Meta-Hebbian (plasticidade adaptativa)
- Predictive Coding (inferência preditiva)
- Active Inference (ação epistêmica)
- VQ-VAE / MONOLITH (quantização)
- Mycelial Reasoning (raciocínio associativo)

Este módulo resolve os conflitos de integração documentados:
- Conflito Estrutural: Denso vs Esparso (adaptador sparse)
- Conflito Dimensional: 32D vs 384D (isomorfismo preservado)

Autor: G (Alexandria Project)
Versão: 1.0
Data: 05/12/2025
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import pickle
import time
from enum import Enum, auto

# =============================================================================
# IMPORTS DOS MÓDULOS (com fallbacks)
# =============================================================================

try:
    from meta_hebbian import (
        MetaHebbianPlasticity, 
        PlasticityRule, 
        create_meta_hebbian_system
    )
    HAS_META_HEBBIAN = True
except ImportError:
    HAS_META_HEBBIAN = False
    print("⚠️ meta_hebbian.py não encontrado")

try:
    from predictive_coding import (
        PredictiveCodingNetwork,
        PredictiveCodingConfig,
        create_predictive_coding_system
    )
    HAS_PREDICTIVE_CODING = True
except ImportError:
    HAS_PREDICTIVE_CODING = False
    print("⚠️ predictive_coding.py não encontrado")

try:
    from active_inference import (
        ActiveInferenceAgent,
        ActiveInferenceAlexandria,
        create_active_inference_system,
        Action,
        ActionType,
        ActiveInferenceConfig
    )
    HAS_ACTIVE_INFERENCE = True
except ImportError:
    HAS_ACTIVE_INFERENCE = False
    print("⚠️ active_inference.py não encontrado")

try:
    from profiles import ReasoningProfile, ALL_PROFILES, get_scout_profile
    HAS_PROFILES = True
except ImportError:
    HAS_PROFILES = False
    print("⚠️ profiles.py não encontrado")



# =============================================================================
# PERFIS DE SISTEMA (Resource Management)
# =============================================================================

class SystemProfile(Enum):
    """Perfis de execução para diferentes hardwares"""
    LITE = auto()       # 8GB RAM (Consumer/Laptop) - Pruning agressivo
    BALANCED = auto()   # 16-32GB RAM (Workstation)
    PERFORMANCE = auto() # 64GB+ RAM (Server) - Sem limites

@dataclass
class ResourceLimits:
    """Limites de recursos por perfil"""
    max_memory_mb: int
    max_graph_nodes: int
    pruning_threshold: float
    
    @classmethod
    def from_profile(cls, profile: SystemProfile) -> 'ResourceLimits':
        if profile == SystemProfile.LITE:
            return cls(8192, 50000, 0.05)  # Max 50k nós
        elif profile == SystemProfile.BALANCED:
            return cls(32768, 500000, 0.01)
        else:
            return cls(1024*1024, 10000000, 0.001)




# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class IntegrationConfig:
    """Configuração unificada do sistema integrado"""
    
    # Perfil de Hardware
    profile: SystemProfile = SystemProfile.BALANCED
    
    # Dimensionalidade (CRÍTICO: deve ser 384 para compatibilidade com VQ-VAE)
    embedding_dim: int = 384
    
    # Predictive Coding (isomórfico - mantém 384D)
    pc_hidden_dims: List[int] = field(default_factory=lambda: [384, 384])
    pc_code_dim: int = 384
    pc_num_iterations: int = 5
    
    # Meta-Hebbian
    meta_num_heads: int = 4
    meta_evolution_interval: int = 100
    
    # Active Inference
    ai_state_dim: int = 64
    ai_planning_horizon: int = 5
    
    # Paths
    state_dir: str = "data/integration/"
    
    def __post_init__(self):
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        
        # Ajusta baseado no perfil se necessário (override manual)
        self.resources = ResourceLimits.from_profile(self.profile)
        if self.profile == SystemProfile.LITE:
            self.ai_planning_horizon = 3  # Horizonte menor para economizar
            self.pc_num_iterations = 3



# =============================================================================
# ADAPTADOR ESPARSO PARA META-HEBBIAN
# =============================================================================

class SparseGraphAdapter:
    """
    Adaptador que permite Meta-Hebbian operar sobre grafos esparsos.
    
    Resolve Conflito A: O Meta-Hebbian original assumia matriz densa,
    mas MycelialReasoning usa Dict[Node, Dict[Node, float]].
    
    Este adaptador:
    1. Aceita grafo esparso como input
    2. Extrai apenas conexões ativas
    3. Aplica regras Meta-Hebbian localmente
    4. Atualiza grafo in-place
    """
    
    def __init__(self, meta_hebbian: 'MetaHebbianPlasticity'):
        self.meta = meta_hebbian
        self.update_history: List[Dict] = []
        
    def apply_to_sparse_graph(
        self,
        graph: Dict[int, Dict[int, float]],
        activated_nodes: Set[int],
        activations: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Aplica Meta-Hebbian a um grafo esparso.
        
        Args:
            graph: Grafo esparso {node: {neighbor: weight}}
            activated_nodes: Conjunto de nós ativados neste ciclo
            activations: Valores de ativação por nó (opcional)
            
        Returns:
            stats: Estatísticas da atualização
        """
        if not activated_nodes:
            return {'updates': 0, 'mean_delta': 0.0}
        
        activations = activations or {n: 1.0 for n in activated_nodes}
        
        updates = 0
        total_delta = 0.0
        
        # Itera apenas sobre nós ativados e seus vizinhos
        for node_i in activated_nodes:
            if node_i not in graph:
                continue
                
            act_i = activations.get(node_i, 0.0)
            
            for node_j, old_weight in list(graph[node_i].items()):
                act_j = activations.get(node_j, 0.0)
                
                # Aplica regra ABCD do Meta-Hebbian
                # Δw = η * (A*o_i*o_j + B*o_i + C*o_j + D)
                rule = self.meta.rules[0]  # Usa primeira regra (ou faz cycling)
                
                delta = rule.learning_rate * (
                    rule.A * act_i * act_j +
                    rule.B * act_i +
                    rule.C * act_j +
                    rule.D
                )
                
                # Atualiza peso
                new_weight = old_weight + delta
                new_weight = np.clip(new_weight, 0.0, 1.0)  # Mantém em [0, 1]
                
                graph[node_i][node_j] = new_weight
                
                # Simetria (se grafo não-direcionado)
                if node_j in graph and node_i in graph[node_j]:
                    graph[node_j][node_i] = new_weight
                
                updates += 1
                total_delta += abs(delta)
        
        stats = {
            'updates': updates,
            'mean_delta': total_delta / max(updates, 1),
            'activated_nodes': len(activated_nodes)
        }
        
        self.update_history.append(stats)
        if len(self.update_history) > 1000:
            self.update_history = self.update_history[-500:]
        
        return stats
    
    def get_local_weights(
        self,
        graph: Dict[int, Dict[int, float]],
        center_node: int,
        radius: int = 2
    ) -> np.ndarray:
        """
        Extrai submatriz local do grafo para análise.
        
        Útil para visualização e diagnóstico.
        """
        # BFS para encontrar vizinhança
        visited = {center_node}
        frontier = {center_node}
        
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                if node in graph:
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_frontier.add(neighbor)
            frontier = new_frontier
        
        # Constrói matriz local
        nodes = sorted(visited)
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        
        matrix = np.zeros((n, n))
        for node_i in nodes:
            if node_i in graph:
                for node_j, weight in graph[node_i].items():
                    if node_j in node_idx:
                        matrix[node_idx[node_i], node_idx[node_j]] = weight
        
        return matrix

    def prune_graph(
        self, 
        graph: Dict[int, Dict[int, float]], 
        max_nodes: int,
        threshold: float
    ) -> Dict[str, int]:
        """
        Remove conexões fracas e nós órfãos para economizar memória (LITE MODE).
        """
        initial_nodes = len(graph)
        removed_edges = 0
        
        # 1. Prune Edges (Conexões fracas)
        nodes_to_check = list(graph.keys())
        for node in nodes_to_check:
            original_edges = list(graph[node].items())
            new_edges = {
                n: w for n, w in original_edges 
                if w >= threshold
            }
            removed_edges += (len(original_edges) - len(new_edges))
            graph[node] = new_edges
            
        # 2. Prune Nodes (se exceder limite, remove LRU ou desconectados)
        removed_nodes = 0
        current_nodes = len(graph)
        if current_nodes > max_nodes:
            # Estratégia simples: remove nós sem conexões de saída
            # (Em produção idealmente usaria LRU real)
            to_remove = [n for n, edges in graph.items() if not edges]
            
            # Se ainda precisar remover, remove aleatórios (LITE é brutal)
            if len(to_remove) < (current_nodes - max_nodes):
                deficit = (current_nodes - max_nodes) - len(to_remove)
                random_candidates = list(graph.keys())[:deficit*2] # Amostra
                to_remove.extend(random_candidates[:deficit])
                
            for n in to_remove:
                if n in graph:
                    del graph[n]
                    removed_nodes += 1
        
        return {
            'removed_edges': removed_edges, 
            'removed_nodes': removed_nodes,
            'final_nodes': len(graph)
        }


# =============================================================================
# CAMADA DE PREDICTIVE CODING ISOMÓRFICA
# =============================================================================

class IsomorphicPredictiveCoding:
    """
    Wrapper do Predictive Coding que preserva dimensionalidade.
    
    Resolve Conflito B: PC original comprimia para 32D,
    mas VQ-VAE espera 384D.
    
    Esta versão:
    1. Mantém input_dim == output_dim == 384
    2. Atua como "filtro preditivo" (denoising/prediction)
    3. Pode ser inserido transparentemente no pipeline
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.pc: Optional[PredictiveCodingNetwork] = None
        
        if HAS_PREDICTIVE_CODING:
            pc_config = PredictiveCodingConfig(
                input_dim=config.embedding_dim,      # 384
                hidden_dims=config.pc_hidden_dims,   # [384, 384]
                code_dim=config.pc_code_dim,         # 384 (isomórfico!)
                num_iterations=config.pc_num_iterations
            )
            self.pc = PredictiveCodingNetwork(pc_config)
        
        # Estatísticas
        self.total_processed = 0
        self.total_surprise = 0.0
        
    def process(
        self,
        embedding: np.ndarray,
        learn: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processa embedding através do PC isomórfico.
        
        Args:
            embedding: Vetor 384D do sentence-transformer
            learn: Se True, atualiza pesos do PC
            
        Returns:
            output: Vetor 384D (mesma dimensão!)
            stats: Estatísticas do processamento
        """
        if self.pc is None:
            # Fallback: passa direto
            return embedding, {'bypass': True}
        
        # Normaliza input
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.config.embedding_dim,):
            raise ValueError(f"Expected {self.config.embedding_dim}D, got {embedding.shape}")
        
        if learn:
            result = self.pc.learn_from_input(embedding)
            output = result['code']
            stats = {
                'iterations': result['inference']['iterations'],
                'error': result['inference']['final_error'],
                'converged': result['inference']['converged']
            }
        else:
            output, infer_stats = self.pc.infer(embedding, max_iterations=self.config.pc_num_iterations)
            stats = {
                'iterations': infer_stats['iterations'],
                'error': infer_stats['final_error'],
                'converged': infer_stats['converged']
            }
        
        # Tracking
        self.total_processed += 1
        self.total_surprise += stats['error']
        stats['avg_surprise'] = self.total_surprise / self.total_processed
        
        return output, stats
    
    def get_surprise(self, embedding: np.ndarray) -> float:
        """
        Computa "surpresa" do input sem modificar pesos.
        
        Surpresa alta = input muito diferente do esperado.
        """
        if self.pc is None:
            return 0.0
        
        _, stats = self.pc.infer(embedding, max_iterations=3)
        return stats['final_error']


# =============================================================================
# PIPELINE INTEGRADO
# =============================================================================

class AlexandriaIntegratedPipeline:
    """
    Pipeline completo que integra todos os módulos cognitivos.
    
    Fluxo:
        Text → Embedding(384D) → PC(384D) → VQ-VAE → Indices → Mycelial
                                    ↑                           ↓
                             Meta-Hebbian ←──────────────────────
                                    ↓
                            Active Inference → Actions
    """
    
    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        embedding_model: Optional[Any] = None,
        vqvae: Optional[Any] = None,
        mycelial: Optional[Any] = None
    ):
        self.config = config or IntegrationConfig()
        
        # Componentes externos (injetados)
        self.embedding_model = embedding_model
        self.vqvae = vqvae
        self.mycelial = mycelial
        
        # Módulos internos
        self._init_modules()
        
        # Estado
        self.observation_count = 0
        self.last_activated: Set[int] = set()
        self.pipeline_stats: List[Dict] = []
        
    def _init_modules(self):
        """Inicializa módulos cognitivos"""
        
        # 1. Predictive Coding (isomórfico)
        self.pc = IsomorphicPredictiveCoding(self.config)
        print(f"✅ Predictive Coding: {self.config.embedding_dim}D → {self.config.pc_code_dim}D")
        
        # 2. Meta-Hebbian + Adaptador Esparso
        self.meta_hebbian = None
        self.sparse_adapter = None
        if HAS_META_HEBBIAN:
            self.meta_hebbian = create_meta_hebbian_system(
                num_codes=1024,  # Tamanho do codebook VQ-VAE
                num_heads=self.config.meta_num_heads
            )
            self.sparse_adapter = SparseGraphAdapter(self.meta_hebbian)
            print(f"✅ Meta-Hebbian: {self.config.meta_num_heads} heads + Sparse Adapter")
        
        # 3. Active Inference
        self.active_inference = None
        if HAS_ACTIVE_INFERENCE:
            self.active_inference = create_active_inference_system(
                state_dim=self.config.ai_state_dim,
                load_existing=True,
                use_predictive_coding=False  # Já temos PC separado
            )
            print(f"✅ Active Inference: state_dim={self.config.ai_state_dim}")
    
    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================
    
    def process_text(
        self,
        text: str,
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Processa texto através do pipeline completo.
        
        Args:
            text: Texto para processar
            learn: Se True, atualiza todos os módulos
            
        Returns:
            result: Dicionário com todos os outputs intermediários
        """
        result = {
            'text_length': len(text),
            'timestamp': time.time(),
            'stages': {}
        }
        
        # Stage 1: Embedding
        embedding = self._get_embedding(text)
        result['stages']['embedding'] = {
            'shape': embedding.shape,
            'norm': float(np.linalg.norm(embedding))
        }
        
        # Stage 2: Predictive Coding
        pc_output, pc_stats = self.pc.process(embedding, learn=learn)
        result['stages']['predictive_coding'] = pc_stats
        result['surprise'] = pc_stats.get('error', 0.0)
        
        # Stage 3: VQ-VAE Quantization
        indices = self._quantize(pc_output)
        result['stages']['quantization'] = {
            'indices': indices.tolist() if isinstance(indices, np.ndarray) else indices,
            'num_codes': len(indices) if hasattr(indices, '__len__') else 1
        }
        
        # Stage 4: Mycelial Observation + Meta-Hebbian
        mycelial_stats = self._observe_mycelial(indices, learn=learn)
        result['stages']['mycelial'] = mycelial_stats
        
        # Stage 5: Active Inference Cycle
        if self.active_inference and learn:
            # Projeta para espaço do AI (384D → 64D via média/pooling)
            ai_obs = self._project_to_ai_space(pc_output)
            ai_result = self.active_inference.perception_action_cycle(
                external_observation=ai_obs
            )
            result['stages']['active_inference'] = {
                'action': ai_result['action_taken'],
                'gaps': ai_result['gaps_detected']
            }
        
        # Tracking
        self.observation_count += 1
        self.pipeline_stats.append({
            'obs': self.observation_count,
            'surprise': result['surprise'],
            'indices': result['stages']['quantization']['indices']
        })
        
        return result
    
    def process_embedding(
        self,
        embedding: np.ndarray,
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Processa embedding pré-computado (pula Stage 1).
        """
        result = {
            'timestamp': time.time(),
            'stages': {}
        }
        
        # Valida dimensionalidade
        if embedding.shape[0] != self.config.embedding_dim:
            raise ValueError(
                f"Embedding deve ser {self.config.embedding_dim}D, "
                f"recebeu {embedding.shape[0]}D"
            )
        
        # Stage 2: Predictive Coding
        pc_output, pc_stats = self.pc.process(embedding, learn=learn)
        result['stages']['predictive_coding'] = pc_stats
        result['surprise'] = pc_stats.get('error', 0.0)
        
        # Stage 3: VQ-VAE
        indices = self._quantize(pc_output)
        result['stages']['quantization'] = {
            'indices': indices.tolist() if isinstance(indices, np.ndarray) else indices
        }
        
        # Stage 4: Mycelial
        mycelial_stats = self._observe_mycelial(indices, learn=learn)
        result['stages']['mycelial'] = mycelial_stats
        
        self.observation_count += 1
        return result
    
    # =========================================================================
    # COMPONENTES DO PIPELINE
    # =========================================================================
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding do texto"""
        if self.embedding_model is not None:
            # Usa modelo real
            return self.embedding_model.encode(text)
        else:
            # Fallback: hash determinístico para testes
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self.config.embedding_dim)
            return emb / np.linalg.norm(emb)
    
    def _quantize(self, embedding: np.ndarray) -> np.ndarray:
        """Quantiza embedding via VQ-VAE"""
        if self.vqvae is not None:
            # Usa VQ-VAE real
            if hasattr(self.vqvae, 'encode'):
                return self.vqvae.encode(embedding)
            elif hasattr(self.vqvae, 'quantize'):
                return self.vqvae.quantize(embedding)
        
        # Fallback: quantização simples por bins
        # Mapeia para indices 0-1023 baseado no valor
        bins = np.linspace(-3, 3, 1024)
        indices = np.digitize(embedding[:4], bins)  # Usa primeiras 4 dims
        return np.clip(indices, 0, 1023)
    
    def _observe_mycelial(
        self,
        indices: np.ndarray,
        learn: bool = True
    ) -> Dict[str, Any]:
        """Observa indices no Mycelial e aplica Meta-Hebbian"""
        stats = {'observed': False, 'meta_hebbian_applied': False}
        
        indices_set = set(int(i) for i in indices)
        
        if self.mycelial is not None:
            # Observação real
            if hasattr(self.mycelial, 'observe'):
                self.mycelial.observe(indices)
            stats['observed'] = True
            
            # Aplica Meta-Hebbian ao grafo esparso
            if learn and self.sparse_adapter is not None:
                if hasattr(self.mycelial, 'connections'):
                    meta_stats = self.sparse_adapter.apply_to_sparse_graph(
                        self.mycelial.connections,
                        activated_nodes=indices_set
                    )
                    stats['meta_hebbian'] = meta_stats
                    stats['meta_hebbian_applied'] = True
        
        # Tracking de ativação
        self.last_activated = indices_set
        stats['activated_count'] = len(indices_set)
        
        return stats
    
    def _project_to_ai_space(self, embedding: np.ndarray) -> np.ndarray:
        """
        Projeta embedding 384D para espaço do Active Inference (64D).
        
        Usa pooling simples: divide em 6 chunks de 64 e faz média.
        """
        target_dim = self.config.ai_state_dim
        source_dim = len(embedding)
        
        if source_dim == target_dim:
            return embedding
        
        # Reshape e média
        chunk_size = source_dim // target_dim
        reshaped = embedding[:chunk_size * target_dim].reshape(target_dim, chunk_size)
        projected = reshaped.mean(axis=1)
        
        return projected
    
    # =========================================================================
    # QUERIES E ANÁLISE
    # =========================================================================
    
    def get_recommendation(self) -> Dict[str, Any]:
        """
        Obtém recomendação do Active Inference sobre próxima ação.
        """
        if self.active_inference is None:
            return {'available': False}
        
        return self.active_inference.suggest_exploration()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Status completo do sistema integrado"""
        status = {
            'observation_count': self.observation_count,
            'modules': {
                'predictive_coding': self.pc.pc is not None,
                'meta_hebbian': self.meta_hebbian is not None,
                'sparse_adapter': self.sparse_adapter is not None,
                'active_inference': self.active_inference is not None,
                'vqvae': self.vqvae is not None,
                'mycelial': self.mycelial is not None,
                'embedding_model': self.embedding_model is not None
            },
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'pc_code_dim': self.config.pc_code_dim,
                'ai_state_dim': self.config.ai_state_dim
            }
        }
        
        # Stats do PC
        if self.pc.pc is not None:
            status['predictive_coding'] = {
                'total_processed': self.pc.total_processed,
                'avg_surprise': self.pc.total_surprise / max(self.pc.total_processed, 1)
            }
        
        # Stats do Meta-Hebbian
        if self.sparse_adapter is not None:
            recent = self.sparse_adapter.update_history[-10:]
            if recent:
                status['meta_hebbian'] = {
                    'total_updates': sum(s['updates'] for s in recent),
                    'avg_delta': np.mean([s['mean_delta'] for s in recent])
                }
        
        # Stats do Active Inference
        if self.active_inference is not None:
            status['active_inference'] = self.active_inference.get_system_status()
        
        return status
    
    def get_surprise_for_text(self, text: str) -> float:
        """
        Computa surpresa para um texto sem modificar o sistema.
        
        Útil para:
        - Detectar outliers
        - Priorizar novidades
        - Filtragem
        """
        embedding = self._get_embedding(text)
        return self.pc.get_surprise(embedding)
    
    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Salva estado de todos os módulos"""
        path = path or f"{self.config.state_dir}/integration_state.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'observation_count': self.observation_count,
            'pipeline_stats': self.pipeline_stats[-1000:],
            'config': {
                'embedding_dim': self.config.embedding_dim,
                'pc_code_dim': self.config.pc_code_dim,
                'ai_state_dim': self.config.ai_state_dim
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        # Salva módulos individuais
        if self.pc.pc is not None:
            self.pc.pc.save_state()
        
        if self.meta_hebbian is not None:
            self.meta_hebbian.save_state()
        
        if self.active_inference is not None:
            self.active_inference.agent.save_state()
        
        return path
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Carrega estado salvo"""
        path = path or f"{self.config.state_dir}/integration_state.pkl"
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.observation_count = state.get('observation_count', 0)
            self.pipeline_stats = state.get('pipeline_stats', [])
            
            # Carrega módulos
            if self.pc.pc is not None:
                self.pc.pc.load_state()
            
            if self.meta_hebbian is not None:
                self.meta_hebbian.load_state()
            
            if self.active_inference is not None:
                self.active_inference.agent.load_state()
            
            return True
        except Exception as e:
            print(f"Erro ao carregar estado: {e}")
            return False


# =============================================================================
# MULTI-AGENT ORCHESTRATOR (The Nemesis Core)
# =============================================================================

class MultiAgentOrchestrator:
    """
    Gerencia múltiplos agentes cognitivos operando sobre a mesma memória.
    
    Agents:
    - Scout: Explora rápido, gera novos nós.
    - Judge: Verifica consistência, remove erros.
    - Weaver: Conecta clusters distantes.
    """
    
    def __init__(self, pipeline: AlexandriaIntegratedPipeline):
        self.pipeline = pipeline
        self.agents: Dict[str, ActiveInferenceAgent] = {}
        self.profiles: Dict[str, Any] = {}
        
        # Inicializa agentes padrão
        if HAS_PROFILES and HAS_ACTIVE_INFERENCE:
            self._spawn_agent('scout', ALL_PROFILES['scout']())
            self._spawn_agent('judge', ALL_PROFILES['judge']())
            self._spawn_agent('weaver', ALL_PROFILES['weaver']())
            
    def _spawn_agent(self, name: str, profile: Any):
        """Cria um agente baseado em um perfil"""
        # Converte perfil para config do Active Inference
        ai_config = ActiveInferenceConfig(
            state_dim=self.pipeline.config.ai_state_dim,
            planning_horizon=profile.planning_horizon,
            temperature=profile.temperature,
            risk_weight=profile.risk_weight,
            ambiguity_weight=profile.ambiguity_weight,
            novelty_bonus=profile.novelty_bonus
        )
        
        # Instancia agente diretamente para injetar config customizada
        agent = ActiveInferenceAgent(
            config=ai_config,
            predictive_coding=None  # Shared perception handles this
        )
        self.agents[name] = agent
        self.profiles[name] = profile
        print(f"   🤖 Agent Spawned: {name.upper()}")

    def run_cycle(self, text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Roda um ciclo completo com todos os agentes.
        1. Input (se houver) é processado pelo pipeline (Scout geralmente pega primeiro)
        2. Todos os agentes observam o estado atual
        3. Cada agente propõe ações
        4. Ações são executadas na memória compartilhada
        """
        results = {}
        
        # 1. Processamento de Input (Shared Perception)
        if text_input:
            # Usa o pipeline principal para gerar embedding/indices
            # Isso é "o que o sistema ouviu"
            perception = self.pipeline.process_text(text_input, learn=False)
            shared_indices = perception['stages']['quantization']['indices']
            results['perception'] = perception
        else:
            shared_indices = []
            
        # 2. Ciclo dos Agentes
        for name, agent in self.agents.items():
            profile = self.profiles[name]
            
            # Limite de passos (Resource Management)
            for _ in range(profile.max_steps_per_cycle):
                # Se houver input recente no pipeline, agente observa
                if shared_indices:
                     # Simula observação (simplificado)
                     pass 
                
                # Agente decide
                action, _ = agent.select_action()
                
                # Executa (na memória compartilhada!)
                result = agent.execute_action(
                    action, 
                    alexandria_interface=self.pipeline.mycelial
                )
                
                if name not in results: results[name] = []
                results[name].append(result)
                
                # Se for Judge, pode ter podado o grafo
                if name == 'judge' and self.pipeline.mycelial:
                    # Judge aciona pruning se necessário
                    if hasattr(self.pipeline.mycelial, 'connections'):
                        limits = self.pipeline.config.resources
                        self.pipeline.sparse_adapter.prune_graph(
                            self.pipeline.mycelial.connections,
                            limits.max_graph_nodes,
                            limits.pruning_threshold
                        )

        return results


# =============================================================================
# TESTE DE INTEGRAÇÃO END-TO-END
# =============================================================================


def test_full_stack(verbose: bool = True) -> Dict[str, Any]:
    """
    Teste completo do pipeline integrado.
    
    Valida:
    1. Dimensionalidade em cada estágio
    2. Fluxo de dados
    3. Atualizações de módulos
    4. Ausência de erros
    """
    print("=" * 70)
    print("TESTE DE INTEGRAÇÃO END-TO-END")
    print("=" * 70)
    
    results = {
        'passed': True,
        'stages': {},
        'errors': []
    }
    
    # 1. Criar pipeline
    print("\n📦 Criando pipeline integrado...")
    try:
        config = IntegrationConfig()
        pipeline = AlexandriaIntegratedPipeline(config)
        results['stages']['init'] = {'status': 'OK'}
        print(f"   ✅ Pipeline criado: embedding_dim={config.embedding_dim}")
    except Exception as e:
        results['errors'].append(f"Init failed: {e}")
        results['passed'] = False
        return results
    
    # 2. Testar com embedding dummy
    print("\n🔄 Testando com embedding dummy (384D)...")
    try:
        dummy_embedding = np.random.randn(384).astype(np.float32)
        dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)
        
        result = pipeline.process_embedding(dummy_embedding)
        
        # Valida saída do PC
        pc_stats = result['stages']['predictive_coding']
        assert 'error' in pc_stats, "PC deve retornar erro"
        
        # Valida quantização
        quant_stats = result['stages']['quantization']
        assert 'indices' in quant_stats, "Quantização deve retornar indices"
        
        results['stages']['embedding_test'] = {
            'status': 'OK',
            'pc_error': pc_stats.get('error', 0),
            'indices_count': len(quant_stats.get('indices', []))
        }
        print(f"   ✅ Embedding processado: surprise={pc_stats.get('error', 0):.4f}")
    except Exception as e:
        results['errors'].append(f"Embedding test failed: {e}")
        results['passed'] = False
    
    # 3. Testar com texto
    print("\n📝 Testando com texto...")
    try:
        test_text = "Vector quantization enables efficient compression of neural representations"
        result = pipeline.process_text(test_text)
        
        results['stages']['text_test'] = {
            'status': 'OK',
            'surprise': result.get('surprise', 0),
            'stages_completed': len(result.get('stages', {}))
        }
        print(f"   ✅ Texto processado: {len(test_text)} chars, surprise={result.get('surprise', 0):.4f}")
    except Exception as e:
        results['errors'].append(f"Text test failed: {e}")
        results['passed'] = False
    
    # 4. Testar múltiplos processamentos
    print("\n🔁 Testando batch de 10 observações...")
    try:
        surprises = []
        for i in range(10):
            emb = np.random.randn(384).astype(np.float32)
            result = pipeline.process_embedding(emb)
            surprises.append(result.get('surprise', 0))
        
        results['stages']['batch_test'] = {
            'status': 'OK',
            'mean_surprise': float(np.mean(surprises)),
            'std_surprise': float(np.std(surprises))
        }
        print(f"   ✅ Batch processado: mean_surprise={np.mean(surprises):.4f}")
    except Exception as e:
        results['errors'].append(f"Batch test failed: {e}")
        results['passed'] = False
    
    # 5. Verificar status do sistema
    print("\n📊 Verificando status do sistema...")
    try:
        status = pipeline.get_system_status()
        
        results['stages']['status_check'] = {
            'status': 'OK',
            'observation_count': status['observation_count'],
            'modules_active': sum(status['modules'].values())
        }
        print(f"   ✅ Observações: {status['observation_count']}")
        print(f"   ✅ Módulos ativos: {sum(status['modules'].values())}/{len(status['modules'])}")
    except Exception as e:
        results['errors'].append(f"Status check failed: {e}")
        results['passed'] = False
    
    # 6. Testar persistência
    print("\n💾 Testando persistência...")
    try:
        save_path = pipeline.save_state()
        loaded = pipeline.load_state(save_path)
        
        results['stages']['persistence'] = {
            'status': 'OK' if loaded else 'WARN',
            'path': save_path
        }
        print(f"   ✅ Estado salvo em: {save_path}")
    except Exception as e:
        results['errors'].append(f"Persistence test failed: {e}")
        results['passed'] = False
    
    # 7. Testar recomendação (se Active Inference disponível)
    if pipeline.active_inference is not None:
        print("\n🎯 Testando recomendações...")
        try:
            rec = pipeline.get_recommendation()
            results['stages']['recommendation'] = {
                'status': 'OK',
                'has_suggestions': 'suggested_queries' in rec
            }
            print(f"   ✅ Recomendação disponível")
        except Exception as e:
            results['errors'].append(f"Recommendation test failed: {e}")
    
    # Resumo
    print("\n" + "=" * 70)
    if results['passed']:
        print("✅ TODOS OS TESTES PASSARAM")
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        for err in results['errors']:
            print(f"   - {err}")
    print("=" * 70)
    
    return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_integrated_pipeline(
    embedding_model: Optional[Any] = None,
    vqvae: Optional[Any] = None,
    mycelial: Optional[Any] = None,
    load_existing: bool = True
) -> AlexandriaIntegratedPipeline:
    """
    Factory function para criar pipeline integrado.
    
    Uso:
        from integration_layer import create_integrated_pipeline
        
        # Com componentes Alexandria reais
        pipeline = create_integrated_pipeline(
            embedding_model=sentence_transformer,
            vqvae=monolith_v13,
            mycelial=mycelial_reasoning
        )
        
        # Processa texto
        result = pipeline.process_text("Some scientific text")
        
        # Ou embedding pré-computado
        result = pipeline.process_embedding(embedding)
    """
    config = IntegrationConfig()
    pipeline = AlexandriaIntegratedPipeline(
        config=config,
        embedding_model=embedding_model,
        vqvae=vqvae,
        mycelial=mycelial
    )
    
    if load_existing:
        loaded = pipeline.load_state()
        if loaded:
            print(f"✅ Pipeline carregado: {pipeline.observation_count} observações")
        else:
            print("🌱 Pipeline inicializado fresh")
    
    return pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Roda teste completo
    results = test_full_stack()
    
    print("""
    
ARQUITETURA INTEGRADA:
======================

    Text ─────────────────────────────────────────────────────┐
         │                                                    │
         ▼                                                    │
    ┌─────────────────┐                                       │
    │ Embedding Model │  (all-MiniLM-L6-v2)                   │
    │   384D output   │                                       │
    └────────┬────────┘                                       │
             │                                                │
             ▼                                                │
    ┌─────────────────────────────────────────┐               │
    │        PREDICTIVE CODING                │               │
    │  384D → [384, 384] → 384D (isomórfico)  │  ◄── CORRIGIDO
    │                                         │               │
    │  • Filtra ruído                         │               │
    │  • Computa surpresa                     │               │
    │  • Aprende predições                    │               │
    └────────┬────────────────────────────────┘               │
             │                                                │
             ▼                                                │
    ┌─────────────────┐                                       │
    │     VQ-VAE      │  (MonolithV13)                        │
    │  384D → Indices │                                       │
    └────────┬────────┘                                       │
             │                                                │
             ▼                                                │
    ┌─────────────────────────────────────────┐               │
    │         MYCELIAL REASONING              │               │
    │      (Sparse Graph Structure)           │  ◄── ADAPTADO │
    │                                         │               │
    │  • Armazena conexões                    │               │
    │  • Propaga ativação                     │               │
    │  • Estrutura: Dict[Node, Dict[Node, w]] │               │
    └────────┬────────────────────────────────┘               │
             │                                                │
             ▼                                                │
    ┌─────────────────────────────────────────┐               │
    │          META-HEBBIAN                   │               │
    │    (via Sparse Graph Adapter)           │  ◄── ADAPTADO │
    │                                         │               │
    │  • Regras ABCD evoluem                  │               │
    │  • Opera sobre conexões ativas apenas   │               │
    │  • Atualiza grafo in-place              │               │
    └────────┬────────────────────────────────┘               │
             │                                                │
             ▼                                                │
    ┌─────────────────────────────────────────┐               │
    │         ACTIVE INFERENCE                │               │
    │        (64D internal state)             │               │
    │                                         │               │
    │  • Detecta gaps                         │               │
    │  • Sugere ações                         │               │
    │  • Planeja exploração                   │               │
    └─────────────────────────────────────────┘               │
                                                              │
    ◄─────────────────────────────────────────────────────────┘
                    Feedback Loop


CONFLITOS RESOLVIDOS:
=====================

    ✅ Conflito A (Denso vs Esparso):
       SparseGraphAdapter traduz operações Meta-Hebbian
       para grafos Dict[Node, Dict[Node, float]]
       
    ✅ Conflito B (32D vs 384D):
       Predictive Coding agora é isomórfico (384D → 384D)
       Pode ser inserido transparentemente no pipeline


COMO USAR:
==========

    from integration_layer import create_integrated_pipeline
    
    # Criar pipeline (com ou sem componentes reais)
    pipeline = create_integrated_pipeline(
        embedding_model=model,      # opcional
        vqvae=monolith,            # opcional
        mycelial=mycelial_net      # opcional
    )
    
    # Processar texto
    result = pipeline.process_text("Vector quantization paper...")
    print(f"Surpresa: {result['surprise']}")
    
    # Obter recomendação
    rec = pipeline.get_recommendation()
    print(f"Sugestão: {rec['recommended_action']}")
    
    # Status do sistema
    status = pipeline.get_system_status()
    
    """)
