import os
import requests
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000"
MAX_WORKERS = 10  # Número de threads paralelas

def ingest_file(file_path):
    """Função worker para ingerir um único arquivo"""
    try:
        payload = {
            "file_path": str(file_path.absolute()),
            "type": "GEN"
        }
        response = requests.post(f"{API_URL}/ingest", json=payload, timeout=30)
        return response.status_code == 200, file_path.name, response.text
    except Exception as e:
        return False, file_path.name, str(e)

def ingest_directory(directory_path, recursive=True):
    """Ingere todos os arquivos suportados em um diretório usando paralelismo"""
    path = Path(directory_path)
    if not path.exists():
        print(f"❌ Diretório não encontrado: {directory_path}")
        return

    # Extensões suportadas (Expandido)
    extensions = ['.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg', '.py', '.json', '.csv']
    
    files = []
    print("🔍 Escaneando diretório...")
    if recursive:
        for ext in extensions:
            files.extend(list(path.rglob(f"*{ext}")))
    else:
        for ext in extensions:
            files.extend(list(path.glob(f"*{ext}")))
            
    total_files = len(files)
    print(f"📦 Encontrados {total_files} arquivos para ingestão.")
    
    success_count = 0
    fail_count = 0
    
    print(f"🚀 Iniciando ingestão com {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submeter todas as tarefas
        future_to_file = {executor.submit(ingest_file, f): f for f in files}
        
        # Barra de progresso
        with tqdm(total=total_files, desc="Ingerindo") as pbar:
            for future in as_completed(future_to_file):
                success, name, msg = future.result()
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"❌ Falha em {name}: {msg}")
                
                pbar.update(1)
            
    print(f"\n✅ Concluído! Sucesso: {success_count}, Falhas: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão Massiva Paralela para Prototype Alexandria")
    parser.add_argument("directory", help="Diretório contendo os arquivos")
    parser.add_argument("--no-recursive", action="store_true", help="Não buscar em subpastas")
    parser.add_argument("--workers", type=int, default=10, help="Número de threads paralelas")
    
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers
    
    print("🚀 Iniciando Ingestão Massiva Paralela...")
    ingest_directory(args.directory, recursive=not args.no_recursive)
