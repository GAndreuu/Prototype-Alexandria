
import lancedb
import sys
import os

def check_lancedb():
    print("="*60)
    print("🔍 CHECKING VECTOR STORE (LANCEDB)")
    print("="*60)
    
    db_path = 'c:\\Users\\G\\Desktop\\Alexandria\\data\\lancedb_store'
    
    if not os.path.exists(db_path):
        print(f"❌ Database path not found: {db_path}")
        return

    try:
        db = lancedb.connect(db_path)
        tables = db.table_names()
        print(f"📂 Tables found: {tables}")
        
        total_chunks = 0
        
        for table_name in tables:
            tbl = db.open_table(table_name)
            count = len(tbl)
            print(f"\n📄 Table: '{table_name}'")
            print(f"   - Total Chunks: {count}")
            total_chunks += count
            
            # Show a sample
            if count > 0:
                df = tbl.search().limit(1).to_pandas()
                print(f"   - Sample Interaction:")
                cols = df.columns.tolist()
                print(f"     Columns: {cols}")
                if 'text' in cols:
                     print(f"     Text Preview: {df.iloc[0]['text'][:100]}...")
                if 'metadata' in cols:
                     print(f"     Metadata: {df.iloc[0]['metadata']}")

        print("\n" + "="*60)
        print(f"✅ TOTAL PROCESSED CHUNKS: {total_chunks}")
        print("="*60)

    except Exception as e:
        print(f"❌ Error accessing LanceDB: {e}")

if __name__ == "__main__":
    check_lancedb()
