"""
Quick script to count unique papers in LanceDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.storage import LanceDBStorage

storage = LanceDBStorage()
table = storage.table

# Get all records
print("Loading records...")
all_records = table.to_pandas()

# Count unique sources (papers)
unique_sources = all_records['source'].nunique()
total_chunks = len(all_records)

print(f"\n{'='*60}")
print(f"ALEXANDRIA DATABASE STATISTICS")
print(f"{'='*60}")
print(f"Total chunks: {total_chunks:,}")
print(f"Unique papers: {unique_sources:,}")
print(f"Avg chunks per paper: {total_chunks/unique_sources:.1f}")
print(f"{'='*60}\n")

# Top papers by chunk count
print("TOP 10 PAPERS BY CHUNK COUNT:")
top_papers = all_records['source'].value_counts().head(10)
for i, (paper, count) in enumerate(top_papers.items(), 1):
    paper_name = Path(paper).name
    print(f"{i:2d}. {paper_name[:60]:<60} ({count:,} chunks)")
