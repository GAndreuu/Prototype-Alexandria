import sys
import os
import numpy as np
import lancedb
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_embeddings(output_path="data/training_embeddings.npy"):
    """
    Exports all embeddings from LanceDB to a numpy file.
    """
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    if not os.path.exists(db_path):
        logger.error(f"LanceDB not found at {db_path}")
        return

    try:
        db = lancedb.connect(db_path)
        table_name = "semantic_memory"
        
        if table_name not in db.table_names():
            logger.error(f"Table {table_name} not found in database")
            return

        table = db.open_table(table_name)
        total_rows = table.count_rows()
        logger.info(f"Found {total_rows} rows in {table_name}")

        # Export in batches to avoid memory issues
        batch_size = 10000
        shard_arrays = []
        
        print("Streaming vectors via Arrow (Offset/Limit)...", flush=True)
        
        try:
            total_batches = (total_rows // batch_size) + 1
            print(f"Total batches expected: {total_batches}", flush=True)
            
            for i in range(0, total_rows, batch_size):
                batch_num = (i // batch_size) + 1
                print(f"Processing batch {batch_num} (Offset {i})...", end="", flush=True)
                
                # Fetch batch via search query
                arrow_batch = table.search().limit(batch_size).offset(i).to_arrow()
                
                # Check if empty (end of data)
                if len(arrow_batch) == 0:
                    print(" Empty.", flush=True)
                    break

                # Extract vectors
                # Using to_pylist() on a small batch (10k) is acceptable, but let's try to be efficient
                # arrow_batch["vector"] is a list<float> column.
                # To numpy:
                # We can access the values array if it's a generic ListArray
                # But search() might return FixedSizeList or Variable List.
                # Let's inspect type on first batch
                vec_col = arrow_batch["vector"]
                
                try:
                    # Try optimized flatten
                    # For ListArray or FixedSizeListArray
                    flat_values = vec_col.values.to_numpy()
                    
                    if hasattr(vec_col.type, 'list_size'):
                         dim = vec_col.type.list_size
                    else:
                         # Fallback for variable list or if list_size not available on type object directly
                         # Assume all have same length if we are training
                         dim = 384 # Default or infer
                         # Verify length of first item
                         if len(vec_col) > 0:
                             dim = len(vec_col[0].as_py())
                    
                    shaped_batch = flat_values.reshape(-1, dim)
                    
                except Exception as e_convert:
                    # Fallback to slower convert if optimized fails (e.g. if chunks are weird)
                    # 10k items is not too bad for python list
                    # print(f" (fallback {e_convert})", end="")
                    vectors_list = vec_col.to_pylist()
                    shaped_batch = np.array(vectors_list, dtype=np.float32)

                shard_arrays.append(shaped_batch)
                print(f" Done. Shape: {shaped_batch.shape}", flush=True)
                
        except Exception as e_iter:
            print(f"\nError in loop: {e_iter}", flush=True)
            import traceback
            traceback.print_exc()
            return
            
        print(f"Concatenating {len(shard_arrays)} batches...", flush=True)
        if not shard_arrays:
            print("No batches were processed!", flush=True)
            return

        vectors = np.vstack(shard_arrays)
        
        print(f"Exported shape: {vectors.shape}", flush=True)
        
        np.save(output_path, vectors)
        print(f"Saved embeddings to {output_path}", flush=True)
        
    except Exception as e:
        logger.error(f"Export failed: {e}")

if __name__ == "__main__":
    export_embeddings()
