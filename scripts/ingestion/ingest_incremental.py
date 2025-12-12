#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental Paper Ingestion - Processa APENAS papers NOVOS
============================================================

Verifica quais arquivos já estão no LanceDB (via source path)
e processa apenas os novos.

Autor: Alexandria Project
"""
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import lancedb

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.semantic_memory import SemanticFileSystem
from core.topology.topology_engine import TopologyEngine


def get_indexed_files():
    """Retorna set de arquivos já indexados no LanceDB"""
    try:
        db = lancedb.connect("data/lancedb_store")
        table = db.open_table("semantic_memory")
        
        # Pegar sources únicos
        df = table.to_pandas()
        sources = set(df['source'].unique())
        
        # Extrair paths normalizados
        indexed = set()
        for s in sources:
            if s and isinstance(s, str):
                # Normalizar path
                p = Path(s)
                indexed.add(p.name.lower())  # Só o nome do arquivo
        
        return indexed
    except Exception as e:
        print(f"[WARN] Erro ao ler LanceDB: {e}")
        return set()


def ingest_file_worker(file_path):
    """Worker function to ingest a single file"""
    try:
        topology = TopologyEngine()
        sfs = SemanticFileSystem(topology)
        
        doc_type = "SCI" if "arxiv" in str(file_path).lower() else "GEN"
        chunks = sfs.index_file(str(file_path), doc_type=doc_type)
        
        return (True, str(file_path), chunks)
    except Exception as e:
        return (False, str(file_path), str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("="*80)
    print("🔄 INCREMENTAL INGESTION - Processa apenas NOVOS papers")
    print(f"   Diretório: {args.directory}")
    print(f"   Workers: {args.workers}")
    print("="*80)

    path = Path(args.directory)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    # Pegar arquivos já indexados
    print("\n📚 Verificando papers já indexados no LanceDB...")
    indexed_files = get_indexed_files()
    print(f"   → {len(indexed_files)} papers já processados")

    # Escanear arquivos
    print("\n🔍 Escaneando diretório...")
    all_files = []
    for ext in ['.pdf', '.txt', '.md']:
        all_files.extend(list(path.rglob(f"*{ext}")))
    
    print(f"   → {len(all_files)} arquivos encontrados")

    # Filtrar apenas novos
    new_files = []
    for f in all_files:
        if f.name.lower() not in indexed_files:
            new_files.append(f)
    
    print(f"\n✨ Papers NOVOS para processar: {len(new_files)}")
    print(f"   (Pulando {len(all_files) - len(new_files)} já indexados)")
    
    if not new_files:
        print("\n✅ Nenhum paper novo! Tudo já está indexado.")
        return

    # Inicializar DB
    print("\n🚀 Iniciando processamento...")
    try:
        top = TopologyEngine()
        sfs = SemanticFileSystem(top)
    except Exception as e:
        print(f"Init failed: {e}")
        return

    start_time = time.time()
    success_count = 0
    fail_count = 0
    total_chunks = 0

    # Processar com multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(ingest_file_worker, f): f for f in new_files}
        
        for i, future in enumerate(as_completed(future_to_file)):
            file_p = future_to_file[future]
            try:
                success, name, result = future.result()
                if success:
                    success_count += 1
                    total_chunks += result
                    status = f"✅ ({result} chunks)"
                else:
                    fail_count += 1
                    status = f"❌ {result[:50]}"
                
                # Progress
                if (i+1) % 10 == 0 or not success:
                    print(f"[{i+1}/{len(new_files)}] {Path(name).name[:50]} - {status}")
                    
            except Exception as e:
                print(f"Worker Error {file_p}: {e}")
                fail_count += 1

    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ INGESTION COMPLETE")
    print(f"   ⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"   📄 Papers processados: {success_count}/{len(new_files)}")
    print(f"   📦 Chunks criados: {total_chunks}")
    print(f"   ❌ Falhas: {fail_count}")
    print("="*80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
