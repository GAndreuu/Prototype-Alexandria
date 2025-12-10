"""
Alexandria :: MycelialReasoning

Grafo de conexões entre códigos do codebook VQ-VAE.
Cresce com uso (Hebbian). Decai com desuso.
Permite propagação de ativação e síntese guiada.

Refatorado para usar Grafo Esparso (Topologia Precisa).

Autor: Alexandria Team
Versão: 2.0 (Sparse Graph)
"""

from __future__ import annotations
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import pickle
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)

# Tipo para nó do grafo: (head_index, code_index)
Node = Tuple[int, int]

@dataclass
class MycelialConfig:
    """Configuração do módulo micelial."""
    num_heads: int = 4
    codebook_size: int = 256
    learning_rate: float = 0.1      # eta
    decay_rate: float = 0.99        # decay factor (mais agressivo para manter esparsidade)
    min_weight: float = 0.1         # threshold para poda (mais alto)
    propagation_steps: int = 2
    activation_threshold: float = 0.05
    save_path: str = "data/mycelial_state.pkl"


class MycelialReasoning:
    """
    Rede micelial de co-ativação entre códigos discretos do VQ-VAE.
    
    Implementação: Grafo Esparso.
    - Nós: (head, code)
    - Arestas: Peso da conexão Hebbiana
    """
    
    def __init__(self, config: Optional[MycelialConfig] = None):
        self.config = config or MycelialConfig()
        self.c = self.config  # alias
        
        # Inicializar estado
        self._init_state()
        
        # Tentar carregar estado salvo
        self._load_state()
    
    def _init_state(self):
        """Inicializa estado zerado."""
        # Grafo esparso: para cada nó, um dict de vizinhos -> peso
        # self.graph[node_a][node_b] = weight
        self.graph: Dict[Node, Dict[Node, float]] = defaultdict(lambda: defaultdict(float))
        
        # Estatísticas de uso (para análise)
        self.total_observations = 0
        self.step = 0
        
        # Cache de ativações (opcional, para stats)
        self.node_activation_counts: Dict[Node, int] = defaultdict(int)

    def _node(self, head: int, code: int) -> Node:
        """Helper para criar chave de nó."""
        return (int(head), int(code))

    # =========================================================================
    # OBSERVAÇÃO (Aprendizado Hebbian)
    # =========================================================================
    
    def observe(self, indices: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """
        Observa um colapso (4 índices dos heads) e atualiza conexões.
        
        Args:
            indices: [h1, h2, h3, h4] - índices dos 4 heads
        """
        # Normalizar input
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        indices = np.array(indices).flatten()
        
        if len(indices) != self.c.num_heads:
            if len(indices) % self.c.num_heads == 0 and len(indices) > self.c.num_heads:
                 self.observe_batch(indices.reshape(-1, self.c.num_heads))
                 return
            logger.warning(f"Esperado {self.c.num_heads} índices, recebido {len(indices)}. Ignorando.")
            return
        
        self.step += 1
        self.total_observations += 1
        
        # Identificar nós ativos neste step
        active_nodes: List[Node] = []
        for h, code in enumerate(indices):
            if code < self.c.codebook_size:
                node = self._node(h, code)
                active_nodes.append(node)
                self.node_activation_counts[node] += 1
        
        # Hebbian: "Cells that fire together, wire together"
        # Conecta todos contra todos no conjunto ativo (clique)
        for a, b in combinations(active_nodes, 2):
            # Conexão bidirecional simétrica
            self.graph[a][b] += self.c.learning_rate
            self.graph[b][a] += self.c.learning_rate

    def observe_batch(self, indices_batch: np.ndarray) -> None:
        """Observa múltiplos colapsos."""
        for indices in indices_batch:
            self.observe(indices)

    # =========================================================================
    # DECAIMENTO
    # =========================================================================
    
    def decay(self) -> None:
        """
        Aplica decaimento global e poda conexões fracas.
        Deve ser chamado periodicamente.
        """
        decay_factor = self.c.decay_rate
        min_w = self.c.min_weight
        
        nodes_to_remove = []
        
        # Iterar sobre todos os nós
        for node_a, neighbors in self.graph.items():
            neighbors_to_remove = []
            
            # Iterar sobre vizinhos
            for node_b, weight in neighbors.items():
                new_weight = weight * decay_factor
                
                if new_weight < min_w:
                    neighbors_to_remove.append(node_b)
                else:
                    neighbors[node_b] = new_weight
            
            # Remover vizinhos podados
            for node_b in neighbors_to_remove:
                del neighbors[node_b]
            
            # Se nó ficou isolado, marcar para remoção
            if not neighbors:
                nodes_to_remove.append(node_a)
        
        # Remover nós isolados do grafo
        for node_a in nodes_to_remove:
            del self.graph[node_a]

    # =========================================================================
    # MANIPULAÇÃO EXPLÍCITA (API para Agentes)
    # =========================================================================

    def connect_nodes(self, node_a: Node, node_b: Node, weight_delta: float = 0.1) -> float:
        """
        Cria ou reforça uma conexão explicitamente.
        Retorna o novo peso da aresta.
        """
        # Garantir simetria
        new_weight = self.graph[node_a][node_b] + weight_delta
        self.graph[node_a][node_b] = new_weight
        self.graph[node_b][node_a] = new_weight
        return new_weight

    def get_neighbors(self, node: Node, top_k: int = 10) -> List[Dict]:
        """
        Retorna vizinhos de um nó ordenados por força.
        """
        if node not in self.graph:
            return []

        neighbors = []
        for neighbor, weight in self.graph[node].items():
            neighbors.append({
                'node': neighbor,
                'weight': weight,
                'head': neighbor[0],
                'code': neighbor[1]
            })

        neighbors.sort(key=lambda x: x['weight'], reverse=True)
        return neighbors[:top_k]

    # =========================================================================
    # PROPAGAÇÃO E RACIOCÍNIO
    # =========================================================================
    
    def propagate(
        self, 
        indices: Union[List[int], np.ndarray], 
        steps: Optional[int] = None
    ) -> Dict[Node, float]:
        """
        Propaga ativação pela rede.
        
        Args:
            indices: Códigos iniciais [h0, h1, h2, h3]
            steps: Passos de propagação
            
        Returns:
            Dict[Node, float]: Mapa de ativação final {node: strength}
        """
        steps = steps or self.c.propagation_steps
        
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        indices = np.array(indices).flatten()
        
        # Ativação inicial
        activation: Dict[Node, float] = defaultdict(float)
        frontier: List[Node] = []
        
        for h, code in enumerate(indices):
            if h < self.c.num_heads and code < self.c.codebook_size:
                node = self._node(h, code)
                activation[node] = 1.0
                frontier.append(node)
        
        # Loop de propagação
        for _ in range(steps):
            new_frontier: List[Node] = []
            # Acumulador para o próximo passo (para não interferir no passo atual)
            next_step_activation = activation.copy()
            
            for source_node in frontier:
                source_act = activation[source_node]
                
                # Espalhar para vizinhos
                if source_node in self.graph:
                    for target_node, weight in self.graph[source_node].items():
                        # A ativação flui proporcional ao peso
                        # (Simplificação: sem normalização complexa por enquanto)
                        flow = source_act * weight
                        
                        if flow > 0:
                            if target_node not in next_step_activation:
                                new_frontier.append(target_node)
                            next_step_activation[target_node] += flow
            
            activation = next_step_activation
            frontier = new_frontier
            
            if not frontier:
                break
                
        return activation

    def synthesize(self, activation: Dict[Node, float]) -> List[int]:
        """
        Colapso: Escolhe o código mais forte para cada head.
        """
        # Agrupar ativações por head
        per_head = defaultdict(list)
        for (h, c), act in activation.items():
            if act >= self.c.activation_threshold:
                per_head[h].append((c, act))
        
        result_indices = []
        for h in range(self.c.num_heads):
            candidates = per_head.get(h, [])
            if not candidates:
                # Fallback: se não houver ativação forte, qual usar?
                # No contexto de 'reason', geralmente temos os índices originais.
                # Mas aqui só recebemos o mapa de ativação.
                # Retornar -1 ou 0 para indicar "sem sinal"?
                # Vamos retornar 0 (padrão) ou lidar no método 'reason'
                result_indices.append(0) 
            else:
                # Winner-take-all
                best_code, _ = max(candidates, key=lambda x: x[1])
                result_indices.append(best_code)
                
        return result_indices

    def reason(
        self, 
        indices: Union[List[int], np.ndarray],
        steps: Optional[int] = None
    ) -> Tuple[List[int], Dict[Node, float]]:
        """
        Pipeline completo: propaga e sintetiza.
        Retorna índices refinados. Se um head não tiver sinal forte, mantém o original.
        """
        # Normalizar indices para acesso
        if isinstance(indices, torch.Tensor):
            indices_list = indices.cpu().numpy().flatten().tolist()
        else:
            indices_list = list(np.array(indices).flatten())

        activation = self.propagate(indices_list, steps)
        
        # Síntese com fallback para o original
        per_head = defaultdict(list)
        for (h, c), act in activation.items():
            if act >= self.c.activation_threshold:
                per_head[h].append((c, act))
        
        refined_indices = []
        for h in range(self.c.num_heads):
            candidates = per_head.get(h, [])
            if not candidates:
                # Mantém original
                refined_indices.append(int(indices_list[h]) if h < len(indices_list) else 0)
            else:
                best_code, _ = max(candidates, key=lambda x: x[1])
                refined_indices.append(best_code)
                
        return refined_indices, activation

    # =========================================================================
    # ANÁLISE E STATS
    # =========================================================================
    
    def get_network_stats(self) -> Dict:
        """Estatísticas da rede esparsa."""
        num_nodes = len(self.graph)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        # Como é bidirecional, arestas únicas = num_edges / 2
        
        weights = [w for neighbors in self.graph.values() for w in neighbors.values()]
        mean_weight = sum(weights) / len(weights) if weights else 0.0
        max_weight = max(weights) if weights else 0.0
        
        return {
            'total_observations': self.total_observations,
            'step': self.step,
            'active_nodes': num_nodes,
            'active_edges': num_edges,
            'mean_weight': mean_weight,
            'max_weight': max_weight,
            'density': "Sparse"
        }

    def get_strongest_connections(self, top_k: int = 20) -> List[Dict]:
        """Retorna as conexões mais fortes."""
        edges = []
        seen = set()
        
        for node_a, neighbors in self.graph.items():
            for node_b, weight in neighbors.items():
                # Evitar duplicatas (a->b e b->a são a mesma aresta lógica)
                edge_key = tuple(sorted([node_a, node_b]))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                
                edges.append({
                    'from': node_a,
                    'to': node_b,
                    'strength': weight
                })
        
        edges.sort(key=lambda x: x['strength'], reverse=True)
        return edges[:top_k]

    def get_hub_codes(self, top_k: int = 10) -> List[Dict]:
        """Retorna códigos com mais conexões."""
        hubs = []
        for node, neighbors in self.graph.items():
            degree = len(neighbors)
            total_strength = sum(neighbors.values())
            hubs.append({
                'head': node[0],
                'code': node[1],
                'degree': degree,
                'total_strength': total_strength,
                'activations': self.node_activation_counts[node]
            })
            
        hubs.sort(key=lambda x: x['total_strength'], reverse=True)
        return hubs[:top_k]

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None) -> None:
        """Salva estado usando pickle (grafo)."""
        path = path or self.c.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'graph': dict(self.graph), # Converter defaultdict para dict
            'node_activation_counts': dict(self.node_activation_counts),
            'total_observations': self.total_observations,
            'step': self.step,
            'config': asdict(self.config)
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Estado micelial (Sparse) salvo em {path}")
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {e}")

    def _load_state(self) -> bool:
        """Carrega estado."""
        path = Path(self.c.save_path)
        if not path.exists():
            return False
            
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Verificar se é formato novo
            if 'graph' not in state:
                logger.warning("Formato de estado antigo detectado. Iniciando novo grafo.")
                return False
                
            # Reconstruir defaultdicts
            self.graph = defaultdict(lambda: defaultdict(float))
            for k, v in state['graph'].items():
                self.graph[k].update(v)
                
            self.node_activation_counts = defaultdict(int)
            self.node_activation_counts.update(state.get('node_activation_counts', {}))
            
            self.total_observations = state['total_observations']
            self.step = state['step']
            
            logger.info(f"Estado micelial carregado. {len(self.graph)} nós ativos.")
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao carregar estado: {e}")
            return False

    def reset(self) -> None:
        self._init_state()


# =============================================================================
# INTEGRAÇÃO COM VQ-VAE (Mantida compatível)
# =============================================================================

import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.reasoning.vqvae.model import MonolithV13
from core.reasoning.vqvae.model_wiki import MonolithWiki

class MycelialVQVAE:
    """Wrapper integrating MycelialReasoning with VQ-VAE models."""
    
    def __init__(self, vqvae_model, mycelial_config: Optional[MycelialConfig] = None):
        self.vqvae = vqvae_model
        self.mycelial = MycelialReasoning(mycelial_config)
        
    @classmethod
    def load_default(cls, model_path=None, use_wiki_model=True):
        # (Mesma lógica de carregamento do original)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wiki_path = "data/monolith_v13_wiki_trained.pth"
        old_path = "data/monolith_v13_trained.pth"
        
        if model_path:
            paths_to_try = [(model_path, "custom", "v13", 384, 256)]
        elif use_wiki_model:
            paths_to_try = [
                (wiki_path, "wiki-trained", "wiki", 512, 1024),
                (old_path, "original", "v13", 384, 256)
            ]
        else:
            paths_to_try = [(old_path, "original", "v13", 384, 256)]
            
        for path, desc, mtype, ldim, hdim in paths_to_try:
            if os.path.exists(path):
                try:
                    if mtype == "wiki":
                        model = MonolithWiki(input_dim=384, hidden_dim=512)
                    else:
                        model = MonolithV13(input_dim=384, hidden_dim=hdim, latent_dim=ldim)
                    
                    state = torch.load(path, map_location=device, weights_only=False)
                    model.load_state_dict(state, strict=False)
                    model.to(device)
                    model.eval()
                    return cls(model)
                except Exception:
                    continue
        
        model = MonolithV13(input_dim=384, hidden_dim=256)
        model.to(device)
        model.eval()
        return cls(model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if x.dim() == 1: x = x.unsqueeze(0)
            out = self.vqvae(x)
            return out['indices']

    def observe(self, indices: torch.Tensor) -> None:
        self.mycelial.observe(indices)

    def reason(self, indices: torch.Tensor, steps: Optional[int] = None) -> Tuple[torch.Tensor, Any]:
        # Adaptador: Mycelial agora retorna (list[int], dict)
        # Precisamos converter list[int] para tensor
        indices_np = indices.cpu().numpy()
        batch_size = indices.shape[0] if indices.dim() > 1 else 1
        
        if batch_size == 1:
            new_indices_list, activation = self.mycelial.reason(indices_np, steps)
            return torch.tensor(new_indices_list, device=indices.device).unsqueeze(0), activation
        else:
            # Batch processing (loop simples por enquanto)
            results = []
            activations = []
            for i in range(batch_size):
                idx = indices_np[i]
                new_idx, act = self.mycelial.reason(idx, steps)
                results.append(new_idx)
                activations.append(act)
            return torch.tensor(results, device=indices.device), activations

    def full_pipeline(self, x: torch.Tensor, reason: bool = True) -> Dict:
        indices = self.encode(x)
        self.observe(indices)
        result = {'original_indices': indices}
        if reason:
            new_indices, activation = self.reason(indices)
            result['reasoned_indices'] = new_indices
            result['activation_pattern'] = activation
        return result


if __name__ == "__main__":
    # Teste rápido
    m = MycelialReasoning()
    print("Observando padrões...")
    # Padrão A: [10, 10, 10, 10]
    for _ in range(20): m.observe([10, 10, 10, 10])
    # Padrão B: [20, 20, 20, 20]
    for _ in range(20): m.observe([20, 20, 20, 20])
    
    print("Stats:", m.get_network_stats())
    
    print("\nPropagando [10, 10, 10, 10] (deve manter):")
    res, _ = m.reason([10, 10, 10, 10])
    print("Resultado:", res)
    
    print("\nPropagando [10, 0, 0, 0] (deve completar para 10,10,10,10):")
    # Nota: 0 não tem conexão forte, mas 10 tem.
    # Se 10 (head 0) ativa 10 (head 1,2,3), então deve preencher.
    res, act = m.reason([10, 0, 0, 0]) 
    print("Resultado:", res)
    
    m.save_state()
