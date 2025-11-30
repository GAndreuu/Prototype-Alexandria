import lancedb
from pathlib import Path

db_path = Path("data/lancedb_store")
if db_path.exists():
    try:
        db = lancedb.connect(str(db_path))
        print(f"Tables in {db_path}: {db.table_names()}")
    except Exception as e:
        print(f"Error connecting: {e}")
else:
    print(f"{db_path} does not exist")
