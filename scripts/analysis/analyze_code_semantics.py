#!/usr/bin/env python3
"""
Alexandria :: Análise Semântica de Códigos VQ-VAE

Investiga o significado semântico dos códigos do codebook,
especialmente os hubs (códigos 0 e 255).

Uso:
    python analyze_code_semantics.py                # Analisa códigos 0 e 255
    python analyze_code_semantics.py --code 42      # Analisa código específico
    python analyze_code_semantics.py --top 10       # Analisa top 10 hubs
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Path do projeto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig

# Imports para dados
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


class CodeSemanticAnalyzer:
    """Analisa significado semântico dos códigos VQ-VAE."""
    
    def __init__(self):
        # Carregar rede micelial
        config = MycelialConfig(save_path=str(PROJECT_ROOT / "data" / "mycelial_state.npz"))
        self.mycelial = MycelialReasoning(config)
        
        # Dados
        self.chunks = []
        self.chunk_indices = []
        
    def load_data_from_lancedb(self, limit: int = 10000) -> int:
        """Carrega chunks e seus índices do LanceDB."""
        if not LANCEDB_AVAILABLE:
            print("❌ LanceDB não disponível")
            return 0
        
        try:
            db_path = PROJECT_ROOT / "data" / "lancedb_store"
            if not db_path.exists():
                print(f"❌ LanceDB não encontrado em {db_path}")
                return 0
            
            db = lancedb.connect(str(db_path))
            table_name = "semantic_memory" if "semantic_memory" in db.table_names() else "chunks"
            
            if table_name not in db.table_names():
                print(f"❌ Tabela {table_name} não encontrada")
                return 0
            
            table = db.open_table(table_name)
            df = table.to_pandas().head(limit)
            
            print(f"📚 Carregando {len(df)} chunks do LanceDB...")
            
            # Verificar colunas
            emb_col = 'embedding' if 'embedding' in df.columns else 'vector'
            text_col = 'text' if 'text' in df.columns else 'content'
            
            if emb_col not in df.columns or text_col not in df.columns:
                print(f"❌ Colunas necessárias não encontradas")
                return 0
            
            # Extrair
            embeddings = np.stack(df[emb_col].values)
            self.chunks = df[text_col].tolist()
            
            # Encode embeddings para índices usando VQ-VAE (fallback simples)
            print("🔢 Encodando chunks para índices VQ-VAE...")
            self.chunk_indices = self._encode_embeddings_to_indices(embeddings)
            
            print(f"✅ Carregados {len(self.chunks)} chunks com índices")
            return len(self.chunks)
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _encode_embeddings_to_indices(self, embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Encode embeddings (384D) para índices VQ-VAE (4 códigos).
        
        Fallback: quantização simples por hash se VQ-VAE não disponível.
        """
        indices_list = []
        
        for emb in embeddings:
            # Dividir embedding em 4 partes
            chunk_size = len(emb) // 4
            indices = np.zeros(4, dtype=np.int64)
            
            for i in range(4):
                start = i * chunk_size
                end = start + chunk_size
                chunk = emb[start:end]
                
                # Hash simples: soma normalizada
                chunk_sum = np.sum(chunk)
                chunk_norm = (chunk_sum - np.min(emb)) / (np.max(emb) - np.min(emb) + 1e-8)
                indices[i] = int(chunk_norm * 255)
            
            indices_list.append(indices)
        
        return indices_list
    
    def analyze_code(self, code: int, head: int, max_examples: int = 20) -> Dict:
        """
        Analisa chunks que ativam um código específico.
        
        Args:
            code: Código do codebook (0-255)
            head: Qual head (0-3)
            max_examples: Máximo de exemplos para mostrar
        
        Returns:
            Dict com análise
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISE SEMÂNTICA: Código {code} no Head {head}")
        print(f"{'='*60}")
        
        # Encontrar chunks que ativam esse código
        matching_chunks = []
        for chunk, indices in zip(self.chunks, self.chunk_indices):
            if indices[head] == code:
                matching_chunks.append(chunk)
        
        if not matching_chunks:
            print(f"⚠️  Nenhum chunk encontrado para código {code} no head {head}")
            return {}
        
        print(f"\n📊 {len(matching_chunks)} chunks ativam este código")
        
        # Estatísticas da rede
        activation_count = self.mycelial.activation_counts[head, code]
        print(f"🔢 Ativações registradas na rede: {activation_count:,}")
        
        # Grau de conectividade
        out_degree = np.sum(self.mycelial.connections[head, code, :] > 0.05)
        in_degree = np.sum(self.mycelial.connections[head, :, code] > 0.05)
        print(f"🔗 Conectividade: {out_degree} saídas, {in_degree} entradas")
        
        # Análise de palavras-chave
        print(f"\n🔍 Análise de Conteúdo:")
        word_freq = Counter()
        
        for chunk in matching_chunks:
            # Extrair palavras (simplificado)
            words = chunk.lower().split()
            # Filtrar palavras comuns
            words = [w for w in words if len(w) > 4 and w.isalpha()]
            word_freq.update(words)
        
        print(f"\n📝 Top 15 palavras mais frequentes:")
        for word, count in word_freq.most_common(15):
            print(f"   {word:20s} : {count:4d}x")
        
        # Exemplos de chunks
        print(f"\n📄 Exemplos de chunks (primeiros {min(max_examples, len(matching_chunks))}):")
        for i, chunk in enumerate(matching_chunks[:max_examples], 1):
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            print(f"\n[{i}] {preview}")
        
        # Conexões mais fortes
        print(f"\n🔗 Top 10 conexões SAINDO deste código:")
        out_connections = self.mycelial.connections[head, code, :]
        top_out = np.argsort(out_connections)[-10:][::-1]
        
        for target_code in top_out:
            strength = out_connections[target_code]
            if strength > 0.05:
                print(f"   {code} → {target_code:3d} : força {strength:.4f}")
        
        print(f"\n🔗 Top 10 conexões ENTRANDO neste código:")
        in_connections = self.mycelial.connections[head, :, code]
        top_in = np.argsort(in_connections)[-10:][::-1]
        
        for source_code in top_in:
            strength = in_connections[source_code]
            if strength > 0.05:
                print(f"   {source_code:3d} → {code} : força {strength:.4f}")
        
        return {
            'code': code,
            'head': head,
            'chunk_count': len(matching_chunks),
            'activation_count': activation_count,
            'out_degree': int(out_degree),
            'in_degree': int(in_degree),
            'top_words': word_freq.most_common(15),
            'examples': matching_chunks[:max_examples]
        }
    
    def compare_hub_codes(self, codes: List[int], head: int = 0):
        """Compara múltiplos códigos para encontrar diferenças semânticas."""
        print(f"\n{'='*60}")
        print(f"COMPARAÇÃO DE HUBS - Head {head}")
        print(f"{'='*60}")
        
        code_words = {}
        
        for code in codes:
            matching_chunks = []
            for chunk, indices in zip(self.chunks, self.chunk_indices):
                if indices[head] == code:
                    matching_chunks.append(chunk)
            
            word_freq = Counter()
            for chunk in matching_chunks:
                words = chunk.lower().split()
                words = [w for w in words if len(w) > 4 and w.isalpha()]
                word_freq.update(words)
            
            code_words[code] = set([w for w, _ in word_freq.most_common(30)])
        
        # Encontrar palavras únicas por código
        print(f"\n🔍 Palavras ÚNICAS por código:")
        for code in codes:
            other_codes = [c for c in codes if c != code]
            other_words = set()
            for other_code in other_codes:
                other_words.update(code_words.get(other_code, set()))
            
            unique_words = code_words[code] - other_words
            
            print(f"\nCódigo {code}:")
            if unique_words:
                print(f"   {', '.join(list(unique_words)[:10])}")
            else:
                print(f"   (nenhuma palavra única)")
        
        # Encontrar palavras comuns
        common_words = code_words[codes[0]]
        for code in codes[1:]:
            common_words &= code_words[code]
        
        print(f"\n🔗 Palavras COMUNS a todos os códigos:")
        if common_words:
            print(f"   {', '.join(list(common_words)[:15])}")
        else:
            print(f"   (nenhuma palavra comum)")


def main():
    parser = argparse.ArgumentParser(description="Analisa semântica dos códigos VQ-VAE")
    parser.add_argument("--code", type=int, help="Analisar código específico")
    parser.add_argument("--head", type=int, default=0, help="Qual head (0-3)")
    parser.add_argument("--top", type=int, help="Analisar top N hubs")
    parser.add_argument("--compare", action="store_true", help="Comparar códigos 0 e 255")
    parser.add_argument("--limit", type=int, default=10000, help="Limite de chunks para carregar")
    args = parser.parse_args()
    
    analyzer = CodeSemanticAnalyzer()
    
    # Carregar dados
    count = analyzer.load_data_from_lancedb(limit=args.limit)
    if count == 0:
        print("❌ Não foi possível carregar dados")
        return 1
    
    if args.code is not None:
        # Analisar código específico
        analyzer.analyze_code(args.code, args.head)
    elif args.top:
        # Analisar top N hubs
        hubs = analyzer.mycelial.get_hub_codes(args.top)
        for hub in hubs:
            if hub['head'] == args.head:
                analyzer.analyze_code(hub['code'], hub['head'], max_examples=5)
                print("\n" + "="*60 + "\n")
    elif args.compare:
        # Comparar códigos 0 e 255
        print("\n🔬 INVESTIGAÇÃO: Códigos 0 vs 255")
        analyzer.analyze_code(0, args.head, max_examples=10)
        analyzer.analyze_code(255, args.head, max_examples=10)
        analyzer.compare_hub_codes([0, 255], args.head)
    else:
        # Default: analisar códigos 0 e 255
        analyzer.analyze_code(0, args.head)
        analyzer.analyze_code(255, args.head)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
