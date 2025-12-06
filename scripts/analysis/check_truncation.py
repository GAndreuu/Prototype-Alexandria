import sys
import os
from pathlib import Path
import pandas as pd
import lancedb

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

def check_truncation():
    print("🔍 Connecting to LanceDB...")
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    db = lancedb.connect(db_path)
    table_name = "semantic_memory"
    
    if table_name not in db.table_names():
        print(f"❌ Table '{table_name}' not found.")
        return

    table = db.open_table(table_name)
    total_rows = table.count_rows()
    print(f"📚 Total chunks in database: {total_rows}")
    
    # We can't easily filter by length in SQL/LanceDB query yet (limited support), 
    # so we might need to iterate or fetch batches.
    # However, we can fetch 'content' and 'source' columns to pandas.
    # Given 200k chunks, this might be heavy but manageable for just text columns.
    
    print("📦 Fetching content data (this might take a moment)...")
    # Fetch all data (might be memory intensive, but we need to check all)
    # Using to_arrow() then to_pandas() is usually safer or just to_pandas() without args if supported
    df = table.to_pandas()
    
    print("🕵️ Analyzing for truncation signatures...")
    # Signature: length == 203 and ends with "..."
    # The previous code was: chunk[:200] + "..." if len(chunk) > 200
    
    # Check for the specific truncation signature
    df['is_truncated'] = df['content'].apply(lambda x: len(x) == 203 and x.endswith("..."))
    
    truncated_count = df['is_truncated'].sum()
    truncated_papers = df[df['is_truncated']]['source'].nunique()
    total_papers = df['source'].nunique()
    
    print("\n📊 Truncation Analysis Report:")
    print("=" * 40)
    print(f"Total Chunks:       {total_rows:,}")
    print(f"Truncated Chunks:   {truncated_count:,} ({truncated_count/total_rows*100:.1f}%)")
    print(f"Healthy Chunks:     {total_rows - truncated_count:,}")
    print("-" * 40)
    print(f"Total Papers:       {total_papers:,}")
    print(f"Affected Papers:    {truncated_papers:,} ({truncated_papers/total_papers*100:.1f}%)")
    print("=" * 40)
    
    if truncated_count > 0:
        print("\n⚠️  Significant data loss detected.")
        print("Recommended Action: Re-ingest affected papers or run an in-place update script.")
        
        # Save list of affected papers
        affected_list = df[df['is_truncated']]['source'].unique()
        with open("affected_papers.txt", "w", encoding="utf-8") as f:
            for p in affected_list:
                f.write(f"{p}\n")
        print(f"📝 List of {len(affected_list)} affected papers saved to 'affected_papers.txt'")

if __name__ == "__main__":
    check_truncation()
