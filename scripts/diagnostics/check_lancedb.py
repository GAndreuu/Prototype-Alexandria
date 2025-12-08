import sys
import os

# Adicionar root ao path
sys.path.append(os.getcwd())

try:
    from core.memory.storage import LanceDBStorage
    from config import settings
    import numpy as np
except ImportError as e:
    print(f"Import Error: {e}")
    # Tentar mockar settings se falhar
    class Settings:
        DATA_DIR = "data"
    settings = Settings()

print("="*60)
print("DIAGNÓSTICO LANCEDB - ALEXANDRIA")
print("="*60)

try:
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    print(f"📂 Diretório esperado do DB: {os.path.abspath(db_path)}")
    
    if not os.path.exists(db_path):
        print("❌ PASTA DO DB NÃO EXISTE!")
    else:
        print("✅ Pasta existe.")

    print("\nInicializando Storage...")
    storage = LanceDBStorage()
    print(f"✅ Conectado ao LanceDB em: {storage.db_path}")
    
    count = storage.count()
    print(f"\n📊 Total de Registros: {count}")
    
    if count == 0:
        print("⚠️ BANCO DE DADOS VAZIO. Os papers não foram indexados.")
    else:
        print("✅ Banco populado.")
        
        # Test Search
        print("\n🔍 Testando busca aleatória...")
        dummy_vec = np.random.randn(384).tolist()
        results = storage.search(dummy_vec, limit=3)
        print(f"   Encontrados {len(results)} resultados.")
        
        for r in results:
            content_preview = r['content'][:100].replace('\n', ' ')
            print(f"   - [{r['modality']}] {content_preview}... (Source: {os.path.basename(r['source'])})")
            
except Exception as e:
    print(f"\n❌ ERRO FATAL: {e}")
    import traceback
    traceback.print_exc()

print("="*60)
