
import lancedb
import pandas as pd
import json

def inspect_metadata():
    print("="*60)
    print("🔍 INSPECTING METADATA STRUCTURE")
    print("="*60)
    
    db_path = r"c:\Users\G\Desktop\Alexandria\data\lancedb_store"
    
    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table(db.table_names()[0])
        
        print(f"Table Schema: {tbl.schema}")
        
        # Fetch 5 rows full
        df = tbl.search().limit(5).to_pandas()
        print(f"Columns: {df.columns.tolist()}")
        
        print(f"Sample Row 1: {df.iloc[0].to_dict()}")
        for i, meta in enumerate(df['metadata']):
            print(f"\n--- Item {i+1} ---")
            print(f"Type: {type(meta)}")
            print(f"Raw: {meta}")
            
            if isinstance(meta, str):
                try:
                    parsed = json.loads(meta)
                    print(f"Parsed Keys: {list(parsed.keys())}")
                except Exception as e:
                    print(f"Parsing Failed: {e}")
            elif isinstance(meta, dict):
                print(f"Keys: {list(meta.keys())}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_metadata()
