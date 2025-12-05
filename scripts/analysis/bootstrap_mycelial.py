#!/usr/bin/env python3
"""
Bootstrap Mycelial Network from LanceDB
========================================

Este script treina a rede Mycelial usando os vetores do LanceDB e o encoder FineWeb.
Pipeline simplificado para V3.1 (sem penalização de hubs).

Uso:
    python bootstrap_mycelial.py --db-path ./data/lancedb --batch-size 1000
"""

import argparse
import logging
import pickle
import sys
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_lancedb_vectors(db_path: str, table_name: str = "semantic_memory") -> np.ndarray:
    """
    Carrega todos os vetores do LanceDB.
    
    Returns:
        np.ndarray: Shape (N, 384) com todos os embeddings
    """
    import lancedb
    
    logger.info(f"Conectando ao LanceDB em: {db_path}")
    db = lancedb.connect(db_path)
    
    # Listar tabelas disponíveis
    tables = db.table_names()
    logger.info(f"Tabelas disponíveis: {tables}")
    
    if table_name not in tables:
        # Tentar encontrar a tabela correta
        possible_tables = [t for t in tables if 'memory' in t.lower() or 'semantic' in t.lower()]
        if possible_tables:
            table_name = possible_tables[0]
            logger.warning(f"Tabela '{table_name}' não encontrada. Usando: {table_name}")
        else:
            raise ValueError(f"Nenhuma tabela de memória encontrada. Disponíveis: {tables}")
    
    table = db.open_table(table_name)
    
    # Carregar todos os dados
    logger.info("Carregando vetores...")
    df = table.to_pandas()
    
    # Encontrar coluna de vetores (geralmente 'vector' ou 'embedding')
    vector_col = None
    for col in ['vector', 'embedding', 'embeddings', 'vec']:
        if col in df.columns:
            vector_col = col
            break
    
    if vector_col is None:
        # Procurar coluna que contenha arrays
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                sample = df[col].iloc[0]
                if isinstance(sample, (list, np.ndarray)) and len(sample) > 100:
                    vector_col = col
                    break
    
    if vector_col is None:
        raise ValueError(f"Coluna de vetores não encontrada. Colunas: {df.columns.tolist()}")
    
    logger.info(f"Usando coluna de vetores: {vector_col}")
    
    # Converter para numpy array
    vectors = np.array(df[vector_col].tolist())
    logger.info(f"Carregados {len(vectors)} vetores com dimensão {vectors.shape[1]}")
    
    return vectors


def load_vq_encoder(encoder_path: Optional[str] = None):
    """
    Carrega o encoder VQ (MONOLITH/V11).
    """
    from core.reasoning.vqvae.model_wiki import MonolithWiki
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use provided path or default to FineWeb model
    if encoder_path is None:
        encoder_path = "data/monolith_v3_fineweb.pt"
    
    if not os.path.exists(encoder_path):
        # Fallback to old wiki model if FineWeb not found
        old_path = "data/monolith_v13_wiki_trained.pth"
        if os.path.exists(old_path):
            logger.warning(f"FineWeb model not found at {encoder_path}. Falling back to {old_path}")
            encoder_path = old_path
        else:
            logger.warning(f"Modelo não encontrado em {encoder_path}")
            return None

    logger.info(f"Carregando MonolithWiki de {encoder_path}...")
    model = MonolithWiki(input_dim=384, hidden_dim=512)
    
    # Load checkpoint
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    
    # Handle nested state dict (FineWeb format)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


class SimpleVectorQuantizer:
    """
    Quantizador simples caso o encoder não seja encontrado.
    Usa k-means nos dados para criar codebooks.
    """
    
    def __init__(self, num_heads: int = 4, codebook_size: int = 256):
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.codebooks = None
        self.head_dim = None
    
    def fit(self, vectors: np.ndarray):
        """Treina codebooks com k-means."""
        from sklearn.cluster import MiniBatchKMeans
        
        self.head_dim = vectors.shape[1] // self.num_heads
        self.codebooks = []
        
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_vectors = vectors[:, start:end]
            
            logger.info(f"Treinando codebook para Head {h}...")
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                batch_size=1024,
                n_init=3,
                random_state=42
            )
            kmeans.fit(head_vectors)
            self.codebooks.append(kmeans)
        
        logger.info("Codebooks treinados!")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantiza vetores para índices.
        
        Returns:
            np.ndarray: Shape (N, 4) com índices de cada head
        """
        if self.codebooks is None:
            raise RuntimeError("Codebooks não treinados. Chame fit() primeiro.")
        
        indices = np.zeros((len(vectors), self.num_heads), dtype=np.int32)
        
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_vectors = vectors[:, start:end]
            indices[:, h] = self.codebooks[h].predict(head_vectors)
        
        return indices


class StandaloneMycelialNetwork:
    """
    Implementação standalone da rede Mycelial para caso
    a classe do projeto não esteja disponível.
    """
    
    def __init__(self, num_heads: int = 4, codebook_size: int = 256):
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.learning_rate = 0.01
        self.decay_rate = 0.001
        
        # Matriz de conexões para cada head
        # connections[h][i, j] = força da conexão entre código i e j no head h
        self.connections = [
            np.zeros((codebook_size, codebook_size), dtype=np.float32)
            for _ in range(num_heads)
        ]
        
        # Contagem de ativações
        self.activation_counts = np.zeros((num_heads, codebook_size), dtype=np.int64)
        
        # Total de observações
        self.total_observations = 0
    
    def observe(self, indices: np.ndarray):
        """
        Observa um padrão de códigos e atualiza conexões (Hebbian learning).
        
        Args:
            indices: Array de 4 índices (um por head)
        """
        indices = np.asarray(indices)
        
        # Learning rate base
        base_lr = self.learning_rate
        
        # Atualizar conexões inter-head (códigos que co-ocorrem)
        for i in range(self.num_heads):
            for j in range(self.num_heads):
                if i != j:
                    code_i, code_j = indices[i], indices[j]
                    
                    # Hebbian update
                    self.connections[i][code_i, code_j] += base_lr
        
        # Atualizar conexões intra-head (vizinhança)
        for h in range(self.num_heads):
            code = indices[h]
            
            lr = base_lr * 0.1
            
            for neighbor in range(max(0, code - 5), min(self.codebook_size, code + 6)):
                if neighbor != code:
                    self.connections[h][code, neighbor] += lr
        
        # Atualizar contagens
        for h, code in enumerate(indices):
            self.activation_counts[h, code] += 1
        
        self.total_observations += 1
        
        # Decay periódico
        if self.total_observations % 1000 == 0:
            self._apply_decay()
    
    def _apply_decay(self):
        """Aplica decay em todas as conexões."""
        for h in range(self.num_heads):
            self.connections[h] *= (1 - self.decay_rate)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas da rede."""
        total_connections = 0
        active_connections = 0
        
        for h in range(self.num_heads):
            total_connections += self.codebook_size ** 2
            active_connections += np.sum(self.connections[h] > 0.01)
        
        return {
            'total_observations': self.total_observations,
            'total_possible_connections': total_connections,
            'active_connections': int(active_connections),
            'network_density': active_connections / total_connections,
            'activation_counts': self.activation_counts.tolist(),
        }
    
    def get_state(self) -> dict:
        """Retorna estado completo para serialização."""
        return {
            'connections': [c.tolist() for c in self.connections],
            'activation_counts': self.activation_counts.tolist(),
            'total_observations': self.total_observations,
            'config': {
                'num_heads': self.num_heads,
                'codebook_size': self.codebook_size,
                'learning_rate': self.learning_rate,
                'decay_rate': self.decay_rate,
            }
        }


def train_mycelial_network(
    indices: np.ndarray,
    mycelial_state_path: str
):
    """
    Treina a rede Mycelial com os índices quantizados.
    
    Args:
        indices: Shape (N, 4) com índices de cada head
        mycelial_state_path: Caminho para salvar o estado
    """
    # Tentar importar a classe MycelialReasoning do projeto
    try:
        from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
        
        config = MycelialConfig(
            num_heads=4,
            codebook_size=256,
            learning_rate=0.01,
            decay_rate=0.001,
            propagation_steps=5,
            activation_threshold=0.05
        )
        mycelial = MycelialReasoning(config)
        using_project_class = True
        logger.info("Usando MycelialReasoning do projeto")
        
    except ImportError:
        logger.warning("MycelialReasoning não encontrado. Usando implementação standalone.")
        mycelial = StandaloneMycelialNetwork()
        using_project_class = False
    
    # Treinar rede
    logger.info(f"Treinando rede Mycelial com {len(indices)} observações...")
    
    batch_size = 1000
    num_batches = (len(indices) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(indices))
        batch = indices[start:end]
        
        for obs in batch:
            mycelial.observe(obs)
        
        if (batch_idx + 1) % 10 == 0:
            progress = (batch_idx + 1) / num_batches * 100
            logger.info(f"Progresso: {progress:.1f}% ({end}/{len(indices)})")
    
    # Salvar estado
    logger.info(f"Salvando estado em: {mycelial_state_path}")
    
    if using_project_class:
        # Ensure path is set
        mycelial.c.save_path = mycelial_state_path
        mycelial.save_state()
    else:
        with open(mycelial_state_path, 'wb') as f:
            pickle.dump(mycelial.get_state(), f)
    
    # Retornar estatísticas
    stats = mycelial.get_network_stats() if using_project_class else mycelial.get_stats()
    return stats


def analyze_results(stats: dict, indices: np.ndarray):
    """Analisa e reporta resultados do treinamento."""
    
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS DO BOOTSTRAP")
    logger.info("="*60)
    
    logger.info(f"\nObservações processadas: {stats.get('total_observations', len(indices))}")
    logger.info(f"Conexões ativas: {stats.get('active_connections', 'N/A')}")
    logger.info(f"Densidade da rede: {stats.get('network_density', 0)*100:.2f}%")
    
    # Análise de utilização por head
    logger.info("\nUtilização por Head:")
    for h in range(4):
        unique_codes = len(np.unique(indices[:, h]))
        logger.info(f"  Head {h}: {unique_codes}/256 códigos ({unique_codes/256*100:.1f}%)")
    
    # Top códigos mais frequentes
    logger.info("\nTop 5 códigos mais frequentes por Head:")
    for h in range(4):
        unique, counts = np.unique(indices[:, h], return_counts=True)
        top_indices = np.argsort(counts)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            code = unique[idx]
            freq = counts[idx]
            pct = freq / len(indices) * 100
            logger.info(f"  Head {h} - #{i+1}: Código {code} ({freq} hits, {pct:.1f}%)")
    
    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Mycelial Network from LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python bootstrap_mycelial.py --db-path ./data/lancedb
        """
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default='./data/lancedb_store',
        help='Caminho para o banco LanceDB'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        default='semantic_memory',
        help='Nome da tabela no LanceDB'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./data/mycelial_state.pkl',
        help='Caminho para salvar o estado da rede'
    )
    
    parser.add_argument(
        '--encoder-path',
        type=str,
        default=None,
        help='Caminho para o encoder VQ (opcional)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Tamanho do batch para processamento'
    )
    
    args = parser.parse_args()
    
    # Criar diretório de output se não existir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Carregar vetores do LanceDB
    vectors = load_lancedb_vectors(args.db_path, args.table)
    
    # 2. Carregar ou criar encoder VQ
    encoder = load_vq_encoder(args.encoder_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if encoder is not None:
        # Usar encoder carregado
        logger.info("Usando MonolithWiki carregado...")
        with torch.no_grad():
            # Process in batches to avoid OOM
            indices_list = []
            batch_size = 1000
            for i in range(0, len(vectors), batch_size):
                batch_vecs = vectors[i:i+batch_size]
                batch_tensor = torch.tensor(batch_vecs, dtype=torch.float32).to(device)
                outputs = encoder(batch_tensor)
                indices_list.append(outputs['indices'].cpu().numpy())
            indices = np.concatenate(indices_list, axis=0)
    else:
        logger.info("Treinando quantizador simples nos dados...")
        quantizer = SimpleVectorQuantizer(num_heads=4, codebook_size=256)
        quantizer.fit(vectors)
        indices = quantizer.encode(vectors)
    
    logger.info(f"Gerados {len(indices)} conjuntos de índices")
    
    # 4. Treinar rede Mycelial
    stats = train_mycelial_network(
        indices=indices,
        mycelial_state_path=str(output_path)
    )
    
    # 5. Analisar resultados
    analyze_results(stats, indices)
    
    logger.info(f"\n✓ Bootstrap completo! Estado salvo em: {output_path}")


if __name__ == '__main__':
    main()
