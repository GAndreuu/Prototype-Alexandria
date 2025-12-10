import lancedb
from pathlib import Path
import os

def get_indexed_count():
    try:
        db = lancedb.connect("data/lancedb_store")
        if "semantic_memory" not in db.table_names():
            return 0
        table = db.open_table("semantic_memory")
        df = table.to_pandas()
        sources = set(df['source'].unique())
        return len(sources)
    except Exception as e:
        print(f"Error reading DB: {e}")
        return 0

def get_total_pdfs():
    path = Path("data/library/arxiv")
    if not path.exists():
        return 0
    return len(list(path.rglob("*.pdf"))) + len(list(path.rglob("*.txt")))

def main():
    print("Checking status...")
    total = get_total_pdfs()
    indexed = get_indexed_count()
    remaining = total - indexed
    
    print(f"Total files in library: {total}")
    print(f"Indexed in DB: {indexed}")
    print(f"Remaining: {remaining}")

if __name__ == "__main__":
    main()
