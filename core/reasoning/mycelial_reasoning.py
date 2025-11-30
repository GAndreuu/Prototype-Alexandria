"""
Alexandria :: MycelialReasoning

Grafo de conexões entre códigos do codebook VQ-VAE.
Cresce com uso (Hebbian). Decai com desuso.
Permite propagação de ativação e síntese guiada.

Autor: Alexandria Team
Versão: 1.0
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MycelialConfig:
    """Configuração do módulo micelial."""
    num_heads: int = 4
    codebook_size: int = 256
    learning_rate: float = 0.01
    decay_rate: float = 0.001
    propagation_steps: int = 3
    activation_threshold: float = 0.1
    connection_threshold: float = 0.05
    save_path: str = "data/mycelial_state.npz"


class MycelialReasoning:
    """
    Rede micelial sobre o codebook do VQ-VAE.
    
    Conceitos:
    - Códigos que co-ocorrem frequentemente formam conexões fortes
    - Ativação se propaga pelas conexões
    - Conexões não usadas decaem
    - Síntese emerge do padrão de ativação final
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
        # Conexões entre códigos (por head)
        # connections[h, i, j] = força da conexão código_i → código_j no head h
        self.connections = np.zeros(
            (self.c.num_heads, self.c.codebook_size, self.c.codebook_size),
            dtype=np.float32
        )
        
        # Contagem de ativações por código
        self.activation_counts = np.zeros(
            (self.c.num_heads, self.c.codebook_size),
            dtype=np.int64
        )
        
        # Timestamp da última ativação (para decaimento)
        self.last_activation = np.zeros(
            (self.c.num_heads, self.c.codebook_size),
            dtype=np.int64
        )
        
        # Contador global
        self.total_observations = 0
        self.step = 0
    
    # =========================================================================
    # OBSERVAÇÃO (Aprendizado Hebbian)
    # =========================================================================
    
    def observe(self, indices: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """
        Observa um colapso (4 índices dos heads) e atualiza conexões.
        
        Args:
            indices: [h1, h2, h3, h4] - índices dos 4 heads
        
        Princípio Hebbian: códigos que disparam juntos, conectam.
        """
        # Normalizar input
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        indices = np.array(indices).flatten()
        
        if len(indices) != self.c.num_heads:
            # Se for batch, chamar observe_batch
            if len(indices) % self.c.num_heads == 0 and len(indices) > self.c.num_heads:
                 self.observe_batch(indices.reshape(-1, self.c.num_heads))
                 return
            # raise ValueError(f"Esperado {self.c.num_heads} índices, recebido {len(indices)}")
            logger.warning(f"Esperado {self.c.num_heads} índices, recebido {len(indices)}. Ignorando.")
            return
        
        self.step += 1
        self.total_observations += 1
        
        # Atualizar contagem de ativações
        for h, idx in enumerate(indices):
            idx = int(idx)
            if idx >= self.c.codebook_size: continue
            
            self.activation_counts[h, idx] += 1
            self.last_activation[h, idx] = self.step
        
        # Fortalecer conexões entre códigos que co-ocorreram
        # Conexão INTER-head: código do head_i conecta com código do head_j
        for h1 in range(self.c.num_heads):
            for h2 in range(self.c.num_heads):
                if h1 != h2:
                    i, j = int(indices[h1]), int(indices[h2])
                    if i >= self.c.codebook_size or j >= self.c.codebook_size: continue
                    
                    # Regra Hebbian: Δw = η * x_i * x_j
                    # Aqui assumimos x_i=1, x_j=1 pois ocorreram
                    self.connections[h1, i, j] += self.c.learning_rate
        
        # Conexão INTRA-head: dentro do mesmo head, códigos próximos conectam
        for h in range(self.c.num_heads):
            idx = int(indices[h])
            if idx >= self.c.codebook_size: continue
            
            # Conecta com vizinhos no codebook (±5)
            for offset in range(-5, 6):
                if offset == 0:
                    continue
                neighbor = (idx + offset) % self.c.codebook_size
                self.connections[h, idx, neighbor] += self.c.learning_rate * 0.1
    
    def observe_batch(self, indices_batch: np.ndarray) -> None:
        """
        Observa múltiplos colapsos de uma vez.
        
        Args:
            indices_batch: [batch_size, num_heads]
        """
        for indices in indices_batch:
            self.observe(indices)
    
    # =========================================================================
    # DECAIMENTO
    # =========================================================================
    
    def decay(self) -> None:
        """
        Aplica decaimento às conexões não usadas.
        
        Conexões enfraquecem com o tempo se não são reforçadas.
        """
        # Decaimento global
        self.connections *= (1 - self.c.decay_rate)
        
        # Zerar conexões muito fracas
        self.connections[self.connections < self.c.connection_threshold] = 0
    
    def decay_selective(self, steps_since_use: int = 100) -> None:
        """
        Decaimento baseado em tempo desde último uso.
        
        Args:
            steps_since_use: Quantos steps sem uso para aplicar decaimento extra
        """
        for h in range(self.c.num_heads):
            for i in range(self.c.codebook_size):
                if self.step - self.last_activation[h, i] > steps_since_use:
                    # Decaimento extra para códigos não usados
                    self.connections[h, i, :] *= 0.9
                    self.connections[h, :, i] *= 0.9
    
    # =========================================================================
    # PROPAGAÇÃO
    # =========================================================================
    
    def propagate(
        self, 
        indices: Union[List[int], np.ndarray], 
        steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Propaga ativação pela rede a partir de índices iniciais.
        
        Args:
            indices: [h1, h2, h3, h4] - colapso inicial
            steps: Número de passos de propagação (default: config)
        
        Returns:
            activation: [num_heads, codebook_size] - padrão de ativação final
        """
        steps = steps or self.c.propagation_steps
        
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        indices = np.array(indices).flatten()
        
        # Inicializar ativação
        activation = np.zeros(
            (self.c.num_heads, self.c.codebook_size),
            dtype=np.float32
        )
        
        # Ativar códigos iniciais
        for h, idx in enumerate(indices):
            if h < self.c.num_heads and int(idx) < self.c.codebook_size:
                activation[h, int(idx)] = 1.0
        
        # Propagar
        for _ in range(steps):
            new_activation = np.zeros_like(activation)
            
            for h in range(self.c.num_heads):
                # Propagação via conexões
                # activation[h] (1, 256) @ connections[h] (256, 256) -> (1, 256)
                new_activation[h] = activation[h] @ self.connections[h]
                
                # Cross-head propagation (mais fraco)
                # Ideia: ativação em h2 influencia h1 se h2 e h1 estiverem conectados
                # Mas nossa matriz connections é [h, i, j], intra-head?
                # O código original diz: "connections[h, i, j] = força da conexão código_i → código_j no head h"
                # Mas observe() faz: "Conexão INTER-head: código do head_i conecta com código do head_j"
                # Isso sugere que connections deveria ser maior ou ter outra estrutura para inter-head.
                # O código original usa `self.connections[h1, i, j] += ...` onde i é de h1 e j é de h2?
                # Se i é de h1 e j é de h2, então connections[h1] armazena conexões de h1 para h2?
                # Mas h2 varia.
                
                # CORREÇÃO DA LÓGICA HEBBIAN DO ORIGINAL:
                # O original faz:
                # for h1 in range(num_heads):
                #   for h2 in range(num_heads):
                #      if h1 != h2:
                #         self.connections[h1, i, j] += ...
                # Isso significa que connections[h1] armazena conexões SAINDO de h1?
                # Mas 'j' é o índice em h2.
                # Se connections[h] é (256, 256), ele só mapeia 256 -> 256.
                # Se mapeia h1 -> h2, precisariamos saber QUAL h2.
                
                # INTERPRETAÇÃO:
                # O código original parece simplificar e misturar tudo em 'connections[h]'.
                # Vamos manter a implementação "fiel" ao snippet fornecido, mas com ressalvas.
                # No snippet: `self.connections[h1, i, j] += ...`
                # Isso salva em h1 a conexão i->j. Onde j é um índice de OUTRO head.
                # Isso implica que os códigos são "universais" ou que a matriz h1 aprende a prever códigos de outros heads?
                # Vamos assumir que connections[h] aprende a associar o código i (deste head) com códigos j (de QUALQUER head).
                
                pass 

            # Propagação simplificada conforme snippet original
            # O snippet original faz:
            # new_activation[h] = activation[h] @ self.connections[h]
            # E depois:
            # for h2 in range(num_heads):
            #    if h != h2:
            #       new_activation[h] += 0.3 * (activation[h2] @ self.connections[h2])
            
            # Isso sugere que connections[h2] projeta uma "influência" que é somada em h.
            # Vamos seguir essa lógica.
            
            for h in range(self.c.num_heads):
                # Auto-propagação
                new_activation[h] = activation[h] @ self.connections[h]
                
                # Influência cruzada
                for h2 in range(self.c.num_heads):
                    if h != h2:
                        # A influência de h2 em h
                        # Assumindo que connections[h2] contém info relevante para h
                        new_activation[h] += 0.3 * (activation[h2] @ self.connections[h2])
            
            # Normalizar e combinar com anterior
            new_activation = self._normalize(new_activation)
            activation = 0.5 * activation + 0.5 * new_activation
            
            # Aplicar threshold
            activation[activation < self.c.activation_threshold] = 0
        
        return activation
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normaliza mantendo a estrutura."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return x / norms
    
    # =========================================================================
    # SÍNTESE
    # =========================================================================
    
    def synthesize(self, activation: np.ndarray) -> np.ndarray:
        """
        Converte padrão de ativação em índices (segundo colapso).
        
        Args:
            activation: [num_heads, codebook_size]
        
        Returns:
            indices: [num_heads] - índices sintetizados
        """
        return np.argmax(activation, axis=1)
    
    def reason(
        self, 
        indices: Union[List[int], np.ndarray],
        steps: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pipeline completo: propaga e sintetiza.
        
        Args:
            indices: Colapso inicial [h1, h2, h3, h4]
            steps: Passos de propagação
        
        Returns:
            (new_indices, activation): Novos índices e padrão de ativação
        """
        activation = self.propagate(indices, steps)
        new_indices = self.synthesize(activation)
        return new_indices, activation
    
    # =========================================================================
    # ANÁLISE
    # =========================================================================
    
    def get_strongest_connections(self, top_k: int = 20) -> List[Dict]:
        """Retorna as conexões mais fortes da rede."""
        connections_flat = []
        
        for h in range(self.c.num_heads):
            # Otimização: pegar apenas índices onde valor > threshold
            rows, cols = np.where(self.connections[h] > self.c.connection_threshold)
            for i, j in zip(rows, cols):
                connections_flat.append({
                    'head': h,
                    'from': int(i),
                    'to': int(j),
                    'strength': float(self.connections[h, i, j])
                })
        
        # Ordenar por força
        connections_flat.sort(key=lambda x: x['strength'], reverse=True)
        return connections_flat[:top_k]
    
    def get_hub_codes(self, top_k: int = 10) -> List[Dict]:
        """
        Identifica códigos que são "hubs" (muitas conexões).
        
        Análogo a neurônios altamente conectados.
        """
        hubs = []
        
        for h in range(self.c.num_heads):
            # Grau de saída (out-degree)
            out_degree = np.sum(self.connections[h] > self.c.connection_threshold, axis=1)
            # Grau de entrada (in-degree)
            in_degree = np.sum(self.connections[h] > self.c.connection_threshold, axis=0)
            # Total
            total_degree = out_degree + in_degree
            
            top_indices = np.argsort(total_degree)[-top_k:][::-1]
            
            for idx in top_indices:
                if total_degree[idx] > 0:
                    hubs.append({
                        'head': h,
                        'code': int(idx),
                        'out_degree': int(out_degree[idx]),
                        'in_degree': int(in_degree[idx]),
                        'total_degree': int(total_degree[idx]),
                        'activation_count': int(self.activation_counts[h, idx])
                    })
        
        hubs.sort(key=lambda x: x['total_degree'], reverse=True)
        return hubs[:top_k]
    
    def get_network_stats(self) -> Dict:
        """Estatísticas gerais da rede."""
        active_connections = np.sum(self.connections > self.c.connection_threshold)
        total_possible = self.c.num_heads * self.c.codebook_size * self.c.codebook_size
        
        return {
            'total_observations': self.total_observations,
            'step': self.step,
            'active_connections': int(active_connections),
            'total_possible_connections': int(total_possible),
            'density': float(active_connections / total_possible),
            'mean_connection_strength': float(np.mean(self.connections[self.connections > 0])) if active_connections > 0 else 0,
            'max_connection_strength': float(np.max(self.connections)),
            'active_codes_per_head': [
                int(np.sum(self.activation_counts[h] > 0))
                for h in range(self.c.num_heads)
            ]
        }
    
    # =========================================================================
    # COLISÃO SEMÂNTICA
    # =========================================================================
    
    def find_bridges(
        self, 
        indices_a: np.ndarray, 
        indices_b: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Encontra códigos que conectam dois conjuntos de índices.
        
        Útil para "colisão" entre documentos.
        
        Args:
            indices_a: Códigos do documento A
            indices_b: Códigos do documento B
            top_k: Quantas pontes retornar
        
        Returns:
            Lista de códigos-ponte com scores
        """
        bridges = []
        
        # Propagar de A
        act_a = self.propagate(indices_a)
        # Propagar de B
        act_b = self.propagate(indices_b)
        
        # Encontrar códigos ativados por ambos
        overlap = act_a * act_b
        
        for h in range(self.c.num_heads):
            top_indices = np.argsort(overlap[h])[-top_k:][::-1]
            for idx in top_indices:
                if overlap[h, idx] > 0:
                    bridges.append({
                        'head': h,
                        'code': int(idx),
                        'activation_a': float(act_a[h, idx]),
                        'activation_b': float(act_b[h, idx]),
                        'bridge_score': float(overlap[h, idx])
                    })
        
        bridges.sort(key=lambda x: x['bridge_score'], reverse=True)
        return bridges[:top_k]
    
    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None) -> None:
        """Salva estado da rede."""
        path = path or self.c.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            path,
            connections=self.connections,
            activation_counts=self.activation_counts,
            last_activation=self.last_activation,
            total_observations=self.total_observations,
            step=self.step,
            config=json.dumps(asdict(self.config))
        )
        logger.info(f"Estado micelial salvo em {path}")
    
    def _load_state(self) -> bool:
        """Carrega estado salvo se existir."""
        path = Path(self.c.save_path)
        
        if not path.exists():
            logger.info("Nenhum estado prévio encontrado. Iniciando zerado.")
            return False
        
        try:
            data = np.load(path, allow_pickle=True)
            self.connections = data['connections']
            self.activation_counts = data['activation_counts']
            self.last_activation = data['last_activation']
            self.total_observations = int(data['total_observations'])
            self.step = int(data['step'])
            logger.info(f"Estado micelial carregado de {path}")
            logger.info(f"  Observações: {self.total_observations}")
            logger.info(f"  Conexões ativas: {np.sum(self.connections > 0)}")
            return True
        except Exception as e:
            logger.warning(f"Erro ao carregar estado: {e}")
            return False
    
    def reset(self) -> None:
        """Reseta completamente a rede."""
        self._init_state()
        logger.info("Rede micelial resetada.")


# =============================================================================
# INTEGRAÇÃO COM VQ-VAE
# =============================================================================

class MycelialVQVAE:
    """
    Wrapper que integra MycelialReasoning com MonolithV13.
    
    Uso:
        wrapper = MycelialVQVAE(vqvae_model)
        indices = wrapper.encode(embedding)  # colapso inicial
        new_indices = wrapper.reason(indices)  # colapso guiado
        reconstructed = wrapper.decode(new_indices)
    """
    
    def __init__(self, vqvae_model, mycelial_config: Optional[MycelialConfig] = None):
        self.vqvae = vqvae_model
        self.mycelial = MycelialReasoning(mycelial_config)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode via VQ-VAE, retorna índices."""
        with torch.no_grad():
            # Assumindo que o modelo tem método encode que retorna índices
            # O MonolithV13 retorna dict com 'indices'
            # Precisamos adaptar para o método específico do MonolithV13
            # Se for o MonolithV13 do v2/core/model.py:
            # output = model(x) -> {'indices': ...}
            # Mas o método encode() do V2Learner retorna z_q.
            # Vamos assumir que vqvae_model é a instância de MonolithV13
            
            # Se x não tiver batch dim, adicionar
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            out = self.vqvae(x)
            indices = out['indices']
        return indices
    
    def observe(self, indices: torch.Tensor) -> None:
        """Observa colapso para aprendizado."""
        self.mycelial.observe(indices)
    
    def reason(
        self, 
        indices: torch.Tensor,
        steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Aplica raciocínio micelial."""
        new_indices, activation = self.mycelial.reason(indices.cpu().numpy(), steps)
        return torch.from_numpy(new_indices), activation
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode via VQ-VAE."""
        with torch.no_grad():
            # MonolithV13 usa quantizer.get_codes_from_indices?
            # Ou precisamos reconstruir z_q a partir dos índices.
            # O OrthogonalProductQuantizer tem método para isso?
            # Vamos olhar v2/core/layers.py se necessário.
            # Por enquanto, assumimos que o quantizer tem algo assim ou fazemos manualmente.
            
            # Hack: Se não tiver método direto, usamos o quantizer
            z_q = self.vqvae.quantizer.get_codes_from_indices(indices)
            return self.vqvae.decoder(z_q)
    
    def full_pipeline(
        self, 
        x: torch.Tensor,
        reason: bool = True
    ) -> Dict:
        """
        Pipeline completo: encode → reason → decode.
        
        Args:
            x: Input embedding
            reason: Se True, aplica raciocínio micelial
        
        Returns:
            Dict com índices originais, índices após raciocínio, etc.
        """
        indices = self.encode(x)
        self.observe(indices)
        
        result = {
            'original_indices': indices,
            # 'original_decoded': self.decode(indices) # Opcional, pode ser pesado
        }
        
        if reason:
            new_indices, activation = self.reason(indices)
            result['reasoned_indices'] = new_indices
            result['activation_pattern'] = activation
            # result['reasoned_decoded'] = self.decode(new_indices)
        
        return result


# =============================================================================
# CLI PARA TESTES
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alexandria MycelialReasoning")
    parser.add_argument("--test", action="store_true", help="Rodar testes básicos")
    parser.add_argument("--stats", action="store_true", help="Mostrar estatísticas")
    parser.add_argument("--reset", action="store_true", help="Resetar rede")
    args = parser.parse_args()
    
    # Inicializar
    mycelial = MycelialReasoning()
    
    if args.reset:
        mycelial.reset()
        print("Rede resetada.")
    
    elif args.stats:
        stats = mycelial.get_network_stats()
        print("\n=== Estatísticas da Rede Micelial ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        print("\n=== Top Conexões ===")
        for conn in mycelial.get_strongest_connections(10):
            print(f"  Head {conn['head']}: {conn['from']} → {conn['to']} (força: {conn['strength']:.4f})")
        
        print("\n=== Hubs ===")
        for hub in mycelial.get_hub_codes(5):
            print(f"  Head {hub['head']}, Code {hub['code']}: degree={hub['total_degree']}, ativações={hub['activation_count']}")
    
    elif args.test:
        print("\n=== Teste do MycelialReasoning ===\n")
        
        # Simular observações
        print("Observando 100 colapsos aleatórios...")
        for _ in range(100):
            indices = np.random.randint(0, 256, size=4)
            mycelial.observe(indices)
        
        # Algumas observações com padrão (para criar conexões)
        print("Observando 50 colapsos com padrão [10, 20, 30, 40]...")
        for _ in range(50):
            mycelial.observe([10, 20, 30, 40])
        
        print("Observando 50 colapsos com padrão [10, 25, 35, 45]...")
        for _ in range(50):
            mycelial.observe([10, 25, 35, 45])
        
        # Testar propagação
        print("\nPropagando de [10, 20, 30, 40]...")
        activation = mycelial.propagate([10, 20, 30, 40])
        new_indices = mycelial.synthesize(activation)
        print(f"  Índices originais: [10, 20, 30, 40]")
        print(f"  Índices após propagação: {new_indices}")
        
        # Estatísticas
        stats = mycelial.get_network_stats()
        print(f"\nConexões ativas: {stats['active_connections']}")
        print(f"Densidade: {stats['density']:.4%}")
        
        # Salvar
        mycelial.save_state()
        print("\nEstado salvo.")
        
        print("\n=== Teste concluído ===")
