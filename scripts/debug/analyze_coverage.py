
import lancedb
import os
import pandas as pd
import json
from pathlib import Path

def analyze_coverage():
    print("="*60)
    print("🔍 ANALYZING PROCESSING COVERAGE")
    print("="*60)
    
    # 1. Get Physical Files
    library_path = Path(r"c:\Users\G\Desktop\Alexandria\data\library\arxiv")
    print(f"\n[1] Scanning Library: {library_path}")
    
    try:
        if not library_path.exists():
             print(f"    ❌ Path does not exist: {library_path}")
             return

        all_files = {f.name for f in library_path.glob("*.pdf")}
        print(f"    found {len(all_files)} PDF files in directory.")
    except Exception as e:
        print(f"    ❌ Error scanning directory: {e}")
        return

    # 2. Get Processed Files from Database
    db_path = r"c:\Users\G\Desktop\Alexandria\data\lancedb_store"
    print(f"\n[2] Scanning Vector Database: {db_path}")
    
    processed_files = set()
    total_chunks = 0
    
    try:
        db = lancedb.connect(db_path)
        tables = db.table_names()
        
        if not tables:
            print("    ❌ No tables found.")
            return
            
        tbl = db.open_table(tables[0])
        total_chunks = len(tbl)
        print(f"    found {total_chunks} total chunks.")
        
        print("    Fetching source column...")
        # Fetching max 300k, should cover all
        df = tbl.search().limit(300000).select(["source"]).to_pandas()
        
        for source in df['source']:
            if not source: continue
            processed_files.add(os.path.basename(str(source)))
                
        print(f"    identified {len(processed_files)} unique documents from chunks.")
        
    except Exception as e:
        print(f"    ❌ Error reading database: {e}")
        return

    # 3. Compare
    print(f"\n[3] Difference Analysis")
    
    # Strict matching first
    unprocessed = all_files - processed_files
    
    # Fuzzy match logic if strict fails significantly
    if len(processed_files) > 0 and len(all_files) > 0:
        
        print("    ℹ️  Verifying using filename stems (ignoring extensions)...")
        processed_stems = {os.path.splitext(f)[0] for f in processed_files}
        all_stems = {os.path.splitext(f)[0] for f in all_files}
        
        # Intersection using stems
        common_stems = all_stems.intersection(processed_stems)
        processed_count = len(common_stems)
        unprocessed_stems = all_stems - processed_stems
        final_processed_count = processed_count
        final_unprocessed_count = len(unprocessed_stems)
        
        # Determine actual unprocessed filenames for reporting
        final_unprocessed_files = []
        for f in all_files:
            if os.path.splitext(f)[0] in unprocessed_stems:
                final_unprocessed_files.append(f)
        
    else:
        final_unprocessed_count = len(unprocessed)
        final_processed_count = len(processed_files)
        final_unprocessed_files = list(unprocessed)

    print(f"\n    📊 SUMMARY:")
    print(f"    Total Physical Files (Library): {len(all_files)}")
    print(f"    Total Processed Docs (DB):      {len(processed_files)} (Chunks: {total_chunks})")
    print(f"    Matched Processed:              {final_processed_count}")
    print(f"    Pending Processing:             {final_unprocessed_count}")
    
    pct = (final_processed_count / len(all_files) * 100) if all_files else 0
    print(f"    Coverage:                       {pct:.1f}%")
    
    if final_unprocessed_count > 0:
        print(f"\n    📄 Sample Unprocessed Files (First 5):")
        for f in final_unprocessed_files[:5]:
            print(f"      - {f}")
            
    if final_unprocessed_count == 0 and len(all_files) > 0:
        print(f"\n    ✅ COMPLETE! All files are in the Vector Database.")

if __name__ == "__main__":
    analyze_coverage()
