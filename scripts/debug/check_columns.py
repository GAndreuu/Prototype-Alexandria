
import lancedb
import pandas as pd

def check_cols():
    db_path = r"c:\Users\G\Desktop\Alexandria\data\lancedb_store"
    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table(db.table_names()[0])
        print(f"COLUMNS: {tbl.schema.names}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_cols()
