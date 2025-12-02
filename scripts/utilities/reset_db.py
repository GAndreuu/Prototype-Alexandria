import shutil
import os
import sys
from pathlib import Path

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

def reset():
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    if os.path.exists(db_path):
        print(f"🗑️ Removendo banco de dados em: {db_path}")
        try:
            shutil.rmtree(db_path)
            print("✅ Banco de dados limpo com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao remover banco: {e}")
    else:
        print("⚠️ Banco de dados não encontrado (já está limpo?).")

if __name__ == "__main__":
    reset()
