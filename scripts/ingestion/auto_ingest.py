import sys
import time
import os
import requests
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuração
API_URL = "http://localhost:8000"
MAGIC_FOLDER = Path("data/magic_folder")

# Garantir que a pasta existe
MAGIC_FOLDER.mkdir(parents=True, exist_ok=True)

class AutoIngestHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Ignorar arquivos temporários ou ocultos
        if file_path.name.startswith('.'):
            return
            
        # Esperar um pouco para garantir que o arquivo foi totalmente copiado
        time.sleep(1)
        
        print(f"✨ Novo arquivo detectado: {file_path.name}")
        self.ingest_file(file_path)

    def ingest_file(self, file_path):
        try:
            payload = {
                "file_path": str(file_path.absolute()),
                "type": "GEN"
            }
            print(f"⏳ Ingerindo {file_path.name}...")
            response = requests.post(f"{API_URL}/ingest", json=payload)
            
            if response.status_code == 200:
                print(f"✅ Sucesso: {file_path.name} foi adicionado à memória!")
            else:
                print(f"❌ Falha ao ingerir {file_path.name}: {response.text}")
                
        except Exception as e:
            print(f"❌ Erro de conexão: {e}")

def start_watching():
    print(f"👀 Monitorando Pasta Mágica: {MAGIC_FOLDER.absolute()}")
    print("📂 Arraste arquivos para esta pasta para ingerir automaticamente.")
    
    event_handler = AutoIngestHandler()
    observer = Observer()
    observer.schedule(event_handler, str(MAGIC_FOLDER), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Verificar se API está online
    try:
        requests.get(f"{API_URL}/docs")
    except:
        print("⚠️ AVISO: O Backend (API) parece estar offline.")
        print("   Por favor, execute 'run_system.bat' primeiro.")
        
    start_watching()
