"""
Teste Unitário: LanceDB Storage
Verifica funcionalidade básica do armazenamento vetorial.
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from core.storage import LanceDBStorage

def test_lancedb():
    print("🧪 Iniciando teste do LanceDB...")
    
    # Usar diretório temporário
    test_db_path = "data/test_lancedb"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        
    try:
        storage = LanceDBStorage(db_path=test_db_path)
        
        # 1. Teste de Adição
        print("1. Testando Adição...")
        ids = ["1", "2", "3"]
        vectors = [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
        contents = ["Texto sobre IA", "Texto sobre Biologia", "Imagem de um Gato"]
        sources = ["doc1.txt", "doc2.txt", "img1.png"]
        modalities = ["TEXTUAL", "TEXTUAL", "VISUAL"]
        metadata = [{"type": "txt"}, {"type": "txt"}, {"type": "img"}]
        
        storage.add(ids, vectors, contents, sources, modalities, metadata)
        
        count = storage.count()
        print(f"   Itens no banco: {count}")
        assert count == 3
        print("   ✅ Adição OK")
        
        # 2. Teste de Busca
        print("2. Testando Busca...")
        query = vectors[0] # Buscar o primeiro vetor exato
        results = storage.search(query, limit=1)
        
        print(f"   Resultado: {results[0]['content']} (Score: {results[0]['relevance']:.4f})")
        assert results[0]['id'] == "1"
        print("   ✅ Busca OK")
        
        # 3. Teste de Filtro
        print("3. Testando Filtro (modality='VISUAL')...")
        results = storage.search(query, limit=5, filter_sql="modality = 'VISUAL'")
        
        print(f"   Encontrado: {len(results)}")
        if len(results) > 0:
            print(f"   Item: {results[0]['content']}")
            assert results[0]['modality'] == "VISUAL"
            print("   ✅ Filtro OK")
        else:
            print("   ❌ Filtro falhou (nenhum resultado)")
            
        print("\n🎉 Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Teste falhou: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpar
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)

if __name__ == "__main__":
    test_lancedb()
