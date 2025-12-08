"""
Clean Ingestion Script
Mass ingests documents into LanceDB using SemanticFileSystem.
Optimized for performance with Multiprocessing.
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
from config import settings

def ingest_file_worker(file_path):
    """Worker function to ingest a single file"""
    try:
        # Initialize isolated SFS for this process
        # We assume TopologyEngine handles its own model loading/sharing check or is lightweight enough
        # Actually SFS init is heavy? It loads models.
        # ProcessPoolExecutor workers initialize once? No.
        # Ideally we initialize SFS once per worker.
        # But for simplicity, we do it here (loading model 8 times is fine for RAM usually).
        
        topology = TopologyEngine()
        sfs = SemanticFileSystem(topology)
        
        # Index
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
    print(f"CLEAN MASS INGESTION - {args.directory}")
    print(f"Workers: {args.workers}")
    print("="*80)

    # Sanity check
    path = Path(args.directory)
    if not path.exists():
        print(f"Directory not found: {path}")
        return

    # Scan files
    print("Scanning files...")
    files = []
    for ext in ['.pdf', '.txt', '.md']:
        files.extend(list(path.rglob(f"*{ext}")))
    
    print(f"Found {len(files)} files to ingest.")
    
    if not files:
        return

    # Initialize DB (create table) once in main process to safely handle schema
    print("Initializing Database...")
    try:
        top = TopologyEngine()
        sfs = SemanticFileSystem(top)
        # Just init check
    except Exception as e:
        print(f"Init failed: {e}")
        return

    start_time = time.time()
    success_count = 0
    fail_count = 0
    total_chunks = 0

    print("\nStarting Ingestion...")
    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(ingest_file_worker, f): f for f in files}
        
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
                    status = f"❌ {result}"
                
                # Progress update
                if (i+1) % 10 == 0 or not success:
                    print(f"[{i+1}/{len(files)}] {Path(name).name} - {status}")
                    
            except Exception as e:
                print(f"Worker Error {file_p}: {e}")
                fail_count += 1

    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Papers: {success_count}/{len(files)}")
    print(f"Chunks: {total_chunks}")
    print(f"Failures: {fail_count}")
    print("="*80)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
