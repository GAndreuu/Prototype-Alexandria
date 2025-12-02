import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.semantic_memory import SemanticFileSystem
from core.topology.topology_engine import TopologyEngine

def ingest_file_direct(sfs, file_path):
    """Ingest a single file using SFS instance"""
    try:
        # Check if already indexed (optional optimization)
        # For now, we rely on SFS to handle duplicates or overwrite
        
        chunks = sfs.index_file(str(file_path))
        return True, file_path.name, f"Indexed {chunks} chunks"
    except Exception as e:
        return False, file_path.name, str(e)

def process_directory(directory_path, workers=4):
    """Process directory directly using SemanticFileSystem"""
    path = Path(directory_path)
    if not path.exists():
        print(f"❌ Directory not found: {directory_path}")
        return

    # Initialize Core Systems
    print("🧠 Initializing Semantic Memory System...")
    topology = TopologyEngine()
    sfs = SemanticFileSystem(topology)
    
    extensions = ['.pdf', '.txt', '.md']
    files = []
    
    print("🔍 Scanning files...")
    for ext in extensions:
        files.extend(list(path.rglob(f"*{ext}")))
        
    total_files = len(files)
    print(f"📦 Found {total_files} files to process.")
    
    success_count = 0
    fail_count = 0
    
    print(f"🚀 Starting direct ingestion with {workers} workers...")
    
    # Pre-load existing files to skip duplicates
    print("📋 Checking existing index...")
    try:
        existing_sources = set(sfs.storage.table.to_pandas()['source'].unique())
        print(f"✅ Found {len(existing_sources)} files already indexed.")
    except Exception as e:
        print(f"⚠️ Could not load existing index: {e}")
        existing_sources = set()

    # Filter files
    files_to_process = []
    for f in files:
        # Normalize path for comparison (LanceDB stores absolute paths usually)
        # We try both absolute and name just in case
        abs_path = str(f.absolute())
        if abs_path not in existing_sources and f.name not in existing_sources:
             files_to_process.append(f)
    
    print(f"📉 Filtered {len(files) - len(files_to_process)} duplicates.")
    print(f"📦 Processing {len(files_to_process)} new files.")
    
    if not files_to_process:
        print("✅ Nothing to do!")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(ingest_file_direct, sfs, f): f for f in files_to_process}
        
        with tqdm(total=len(files_to_process), desc="Ingesting") as pbar:
            for future in as_completed(future_to_file):
                success, name, msg = future.result()
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                
                pbar.update(1)
                
    print(f"\n✅ Completed! Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Mass Ingestion (No API required)")
    parser.add_argument("directory", help="Directory to ingest")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    process_directory(args.directory, workers=args.workers)
