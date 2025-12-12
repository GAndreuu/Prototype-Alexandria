#!/usr/bin/env python3
"""
Alexandria :: Integração MycelialReasoning com Dados Reais

Treina a rede micelial com chunks reais do banco de dados.
Gera relatório de análise da estrutura emergente.

Uso:
    python train_mycelial.py                    # Treina com todos os chunks
    python train_mycelial.py --limit 1000       # Limita quantidade
    python train_mycelial.py --analyze          # Só analisa (não treina)
    python train_mycelial.py --export report    # Exporta relatório
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Adicionar path do projeto
# Adicionar path do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS DO PROJETO
# =============================================================================

def safe_import(module_name: str, class_name: str = None):
    """Import seguro com fallback."""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        logger.warning(f"Não foi possível importar {class_name} de {module_name}: {e}")
        return None


# Tentar imports
MycelialReasoning = safe_import('core.reasoning.mycelial_reasoning', 'MycelialReasoning')
MycelialConfig = safe_import('core.reasoning.mycelial_reasoning', 'MycelialConfig')

# Imports do Alexandria (ajustar conforme estrutura real)
SemanticMemory = safe_import('core.memory.semantic_memory', 'SemanticMemory')
SemanticFileSystem = safe_import('core.memory.semantic_memory', 'SemanticFileSystem')

# Tentar importar VQ-VAE
MonolithV13 = safe_import('core.reasoning.vqvae.model', 'MonolithV13')

# Tentar importar modelo de embedding
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False

# Torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class TrainConfig:
    """Configuração do treinamento."""
    
    # Paths
    data_path: str = "data/semantic_memory.db"
    vqvae_path: str = "data/monolith_v13_trained.pth"
    mycelial_path: str = "data/mycelial_state.npz"
    report_path: str = "reports/mycelial_analysis.json"
    
    # Treinamento
    batch_size: int = 32
    limit: Optional[int] = None
    decay_every: int = 500
    save_every: int = 1000
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384


# =============================================================================
# LOADER DE DADOS
# =============================================================================

class DataLoader:
    """Carrega dados do Alexandria."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.chunks = []
        self.embeddings = []
        
    def load_from_lancedb(self) -> int:
        """Carrega do LanceDB se disponível."""
        try:
            import lancedb
            
            db_path = PROJECT_ROOT / "data" / "lancedb_store"
            if not db_path.exists():
                return 0
                
            db = lancedb.connect(str(db_path))
            
            # Tentar tabela de chunks
            table_name = "semantic_memory" if "semantic_memory" in db.table_names() else "chunks"
            if table_name in db.table_names():
                table = db.open_table(table_name)
                df = table.to_pandas()
                
                # Handle column names
                emb_col = 'embedding' if 'embedding' in df.columns else 'vector'
                text_col = 'text' if 'text' in df.columns else 'content'
                
                if emb_col in df.columns:
                    self.embeddings = np.stack(df[emb_col].values)
                if text_col in df.columns:
                    self.chunks = df[text_col].tolist()
                
                logger.info(f"Carregado {len(self.embeddings)} embeddings do LanceDB (table: {table_name})")
                return len(self.embeddings)
                
        except Exception as e:
            logger.warning(f"Erro ao carregar LanceDB: {e}")
        
        return 0
    
    def load_from_sqlite(self) -> int:
        """Carrega do SQLite se disponível."""
        try:
            import sqlite3
            
            db_path = PROJECT_ROOT / self.config.data_path
            if not db_path.exists():
                return 0
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Tentar buscar chunks com embeddings
            cursor.execute("""
                SELECT text, embedding FROM chunks 
                WHERE embedding IS NOT NULL
                LIMIT ?
            """, (self.config.limit or 999999,))
            
            rows = cursor.fetchall()
            
            for text, emb_blob in rows:
                self.chunks.append(text)
                # Embedding pode estar como blob ou JSON
                if isinstance(emb_blob, bytes):
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                else:
                    emb = np.array(json.loads(emb_blob), dtype=np.float32)
                self.embeddings.append(emb)
            
            if self.embeddings:
                self.embeddings = np.stack(self.embeddings)
                logger.info(f"Carregado {len(self.embeddings)} embeddings do SQLite")
                
            conn.close()
            return len(self.embeddings) if isinstance(self.embeddings, np.ndarray) else 0
            
        except Exception as e:
            logger.warning(f"Erro ao carregar SQLite: {e}")
        
        return 0
    
    def load_from_numpy(self) -> int:
        """Carrega de arquivo numpy se disponível."""
        try:
            emb_path = PROJECT_ROOT / "data" / "embeddings.npy"
            if emb_path.exists():
                self.embeddings = np.load(str(emb_path))
                logger.info(f"Carregado {len(self.embeddings)} embeddings de numpy")
                return len(self.embeddings)
            
            # Check training_embeddings.npy
            train_emb_path = PROJECT_ROOT / "data" / "training_embeddings.npy"
            if train_emb_path.exists():
                self.embeddings = np.load(str(train_emb_path))
                logger.info(f"Carregado {len(self.embeddings)} embeddings de training_embeddings.npy")
                return len(self.embeddings)
                
            # Tentar .npz
            npz_path = PROJECT_ROOT / "data" / "embeddings.npz"
            if npz_path.exists():
                data = np.load(str(npz_path))
                if 'embeddings' in data:
                    self.embeddings = data['embeddings']
                if 'texts' in data:
                    self.chunks = data['texts'].tolist()
                logger.info(f"Carregado {len(self.embeddings)} embeddings de npz")
                return len(self.embeddings)
                
        except Exception as e:
            logger.warning(f"Erro ao carregar numpy: {e}")
        
        return 0
    
    def load(self) -> int:
        """Tenta carregar de qualquer fonte disponível."""
        # Tentar LanceDB primeiro
        count = self.load_from_lancedb()
        if count > 0:
            return count
        
        # Tentar SQLite
        count = self.load_from_sqlite()
        if count > 0:
            return count
        
        # Tentar numpy
        count = self.load_from_numpy()
        if count > 0:
            return count
        
        logger.error("Nenhuma fonte de dados encontrada!")
        return 0
    
    def get_batches(self, batch_size: int):
        """Gera batches de embeddings."""
        if not isinstance(self.embeddings, np.ndarray) or len(self.embeddings) == 0:
            return
        
        n = len(self.embeddings)
        for i in range(0, n, batch_size):
            yield self.embeddings[i:i+batch_size]


# =============================================================================
# ENCODER (VQ-VAE ou fallback)
# =============================================================================

class Encoder:
    """Encapsula encoding para índices."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.vqvae = None
        self.use_fallback = False
        
        self._load_vqvae()
    
    def _load_vqvae(self):
        """Tenta carregar VQ-VAE."""
        if MonolithV13 is None:
            logger.warning("MonolithV13 não disponível, usando fallback")
            self.use_fallback = True
            return
        
        vqvae_path = PROJECT_ROOT / self.config.vqvae_path
        if not vqvae_path.exists():
            logger.warning(f"Modelo VQ-VAE não encontrado em {vqvae_path}, usando fallback")
            self.use_fallback = True
            return
        
        try:
            self.vqvae = MonolithV13()
            state = torch.load(str(vqvae_path), map_location='cpu')
            self.vqvae.load_state_dict(state)
            self.vqvae.eval()
            logger.info("VQ-VAE carregado com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao carregar VQ-VAE: {e}, usando fallback")
            self.use_fallback = True
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Converte embeddings em índices.
        
        Args:
            embeddings: [batch, 384]
            
        Returns:
            indices: [batch, 4]
        """
        if self.use_fallback:
            return self._fallback_encode(embeddings)
        
        with torch.no_grad():
            x = torch.from_numpy(embeddings).float()
            # Assumindo API do MonolithV13
            _, indices, _ = self.vqvae.encode(x)
            return indices.cpu().numpy()
    
    def _fallback_encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fallback: quantização simples por hash.
        
        Não é tão bom quanto VQ-VAE mas permite testar o micelial.
        """
        batch_size = embeddings.shape[0]
        indices = np.zeros((batch_size, 4), dtype=np.int64)
        
        # Dividir embedding em 4 partes e quantizar cada uma
        chunk_size = embeddings.shape[1] // 4
        
        for i in range(4):
            start = i * chunk_size
            end = start + chunk_size
            chunk = embeddings[:, start:end]
            
            # Hash simples: soma → mod 256
            chunk_sum = np.sum(chunk, axis=1)
            # Normalizar para 0-255
            chunk_norm = (chunk_sum - chunk_sum.min()) / (chunk_sum.max() - chunk_sum.min() + 1e-8)
            indices[:, i] = (chunk_norm * 255).astype(np.int64)
        
        return indices


# =============================================================================
# TRAINER
# =============================================================================

class MycelialTrainer:
    """Treina MycelialReasoning com dados reais."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Inicializar componentes
        self.loader = DataLoader(config)
        self.encoder = Encoder(config)
        
        # Inicializar micelial
        mycelial_config = MycelialConfig(
            save_path=str(PROJECT_ROOT / config.mycelial_path)
        )
        self.mycelial = MycelialReasoning(mycelial_config)
        
        # Estatísticas
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_chunks': 0,
            'processed': 0,
            'batches': 0,
        }
    
    def train(self, limit: Optional[int] = None) -> Dict:
        """
        Treina a rede micelial com dados reais.
        
        Returns:
            Dict com estatísticas do treinamento
        """
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Carregar dados
        logger.info("Carregando dados...")
        count = self.loader.load()
        
        if count == 0:
            logger.error("Sem dados para treinar!")
            return self.stats
        
        self.stats['total_chunks'] = count
        
        if limit:
            count = min(count, limit)
        
        logger.info(f"Treinando com {count} chunks...")
        
        # Processar em batches
        for i, batch in enumerate(self.loader.get_batches(self.config.batch_size)):
            if limit and self.stats['processed'] >= limit:
                break
            
            # Encode para índices
            indices = self.encoder.encode(batch)
            
            # Observar cada item do batch
            self.mycelial.observe_batch(indices)
            
            self.stats['processed'] += len(batch)
            self.stats['batches'] = i + 1
            
            # Decay periódico
            if self.stats['processed'] % self.config.decay_every == 0:
                self.mycelial.decay()
            
            # Save periódico
            if self.stats['processed'] % self.config.save_every == 0:
                self.mycelial.save_state()
                logger.info(f"Processado: {self.stats['processed']}/{count}")
        
        # Save final
        self.mycelial.save_state()
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['network'] = self.mycelial.get_network_stats()
        
        logger.info(f"Treinamento concluído: {self.stats['processed']} chunks")
        
        return self.stats
    
    def analyze(self) -> Dict:
        """
        Analisa a rede micelial treinada.
        
        Returns:
            Dict com análise completa
        """
        logger.info("Analisando rede micelial...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'network_stats': self.mycelial.get_network_stats(),
            'top_connections': self.mycelial.get_strongest_connections(50),
            'hubs': self.mycelial.get_hub_codes(20),
        }
        
        # Análise adicional
        stats = analysis['network_stats']
        
        density_val = stats['density']
        is_sparse = True
        if isinstance(density_val, (float, int)):
            is_sparse = density_val < 0.1
        
        analysis['health'] = {
            'is_sparse': is_sparse,
            'has_hubs': len(analysis['hubs']) > 0,
            'has_connections': stats['active_edges'] > 0,
            'observations_sufficient': stats['total_observations'] >= 100,
        }
        
        analysis['health']['overall'] = all(analysis['health'].values())
        
        # Distribuição de grau por head
        analysis['degree_distribution'] = {}
        # Distribuição de grau por head (Sparse)
        analysis['degree_distribution'] = {}
        degrees_per_head = {h: [] for h in range(4)}
        
        # Coletar graus
        for (head, code), neighbors in self.mycelial.graph.items():
             # Contar vizinhos fortes > 0.05
             degree = sum(1 for w in neighbors.values() if w > 0.05)
             if 0 <= head < 4:
                 degrees_per_head[head].append(degree)
        
        for h in range(4):
            degs = degrees_per_head[h]
            if not degs:
                degs = [0]
            
            analysis['degree_distribution'][f'head_{h}'] = {
                'mean': float(np.mean(degs)),
                'max': int(np.max(degs)),
                'nonzero': int(sum(1 for d in degs if d > 0)),
            }
            
        # Active codes per head calculation
        stats['active_codes_per_head'] = [0] * 4
        for (head, code) in self.mycelial.graph.keys():
            if 0 <= head < 4:
                stats['active_codes_per_head'][head] += 1

        # Key mapping for backward compatibility
        stats['mean_connection_strength'] = stats.get('mean_weight', 0.0)
        stats['max_connection_strength'] = stats.get('max_weight', 0.0)
        
        return analysis
    
    def export_report(self, path: Optional[str] = None) -> str:
        """Exporta relatório em JSON."""
        path = path or str(PROJECT_ROOT / self.config.report_path)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'training': self.stats,
            'analysis': self.analyze(),
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Relatório exportado: {path}")
        return path


# =============================================================================
# CLI
# =============================================================================

def print_analysis(analysis: Dict):
    """Imprime análise formatada."""
    print("\n" + "=" * 60)
    print("ANÁLISE DA REDE MICELIAL")
    print("=" * 60)
    
    stats = analysis['network_stats']
    print(f"\n📊 Estatísticas Gerais:")
    print(f"   Observações: {stats['total_observations']:,}")
    print(f"   Conexões ativas: {stats['active_edges']:,}")
    if isinstance(stats['density'], (float, int)):
        print(f"   Densidade: {stats['density']:.4%}")
    else:
        print(f"   Densidade: {stats['density']}")
    print(f"   Força média: {stats['mean_connection_strength']:.4f}")
    print(f"   Força máxima: {stats['max_connection_strength']:.4f}")
    
    print(f"\n🔗 Códigos ativos por head:")
    for h, count in enumerate(stats['active_codes_per_head']):
        print(f"   Head {h}: {count}/256 ({100*count/256:.1f}%)")
    
    print(f"\n🏆 Top 10 Hubs:")
    for hub in analysis['hubs'][:10]:
        print(f"   Head {hub['head']}, Code {hub['code']:3d}: "
              f"degree={hub['degree']:3d}, ativações={hub['activations']:,}")
    
    print(f"\n⚡ Top 10 Conexões:")
    for conn in analysis['top_connections'][:10]:
        # Sparse graph uses tuples (head, code)
        h1, c1 = conn['from']
        h2, c2 = conn['to']
        print(f"   H{h1}:{c1} ↔ H{h2}:{c2} (força: {conn['strength']:.4f})")
    
    print(f"\n❤️ Saúde da Rede:")
    health = analysis['health']
    print(f"   Esparsa (<10%): {'✅' if health['is_sparse'] else '❌'}")
    print(f"   Tem hubs: {'✅' if health['has_hubs'] else '❌'}")
    print(f"   Tem conexões: {'✅' if health['has_connections'] else '❌'}")
    print(f"   Observações suficientes: {'✅' if health['observations_sufficient'] else '❌'}")
    print(f"   Status geral: {'✅ SAUDÁVEL' if health['overall'] else '❌ PRECISA ATENÇÃO'}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Treina MycelialReasoning com dados reais")
    parser.add_argument("--limit", type=int, help="Limitar número de chunks")
    parser.add_argument("--analyze", action="store_true", help="Só analisar, não treinar")
    parser.add_argument("--export", type=str, help="Exportar relatório para arquivo")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch")
    args = parser.parse_args()
    
    # Verificar imports
    if MycelialReasoning is None:
        logger.error("MycelialReasoning não disponível!")
        logger.error("Certifique-se de que core/mycelial_reasoning.py existe")
        return 1
    
    # Configurar
    config = TrainConfig(
        batch_size=args.batch_size,
        limit=args.limit,
    )
    
    # Inicializar trainer
    trainer = MycelialTrainer(config)
    
    if args.analyze:
        # Só analisar
        analysis = trainer.analyze()
        print_analysis(analysis)
    else:
        # Treinar
        stats = trainer.train(limit=args.limit)
        
        if stats['processed'] > 0:
            # Analisar após treino
            analysis = trainer.analyze()
            print_analysis(analysis)
    
    # Exportar se pedido
    if args.export:
        trainer.export_report(args.export)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
