import sys
import os
import numpy as np
import lancedb
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        embeddings = []
        
        # LanceDB iterator
        # Note: to_arrow() or to_pandas() might be faster but let's be safe with memory
        # actually table.to_pandas() with columns=['vector'] is efficient enough for 128k rows
        
        logger.info("Loading vectors...")
        # Use to_arrow() which is more robust across versions
        arrow_table = table.to_arrow()
        
        # Extract vector column as numpy array
        # Arrow stores list<float> which converts to object array of numpy arrays
        # We need to stack them
        vectors_list = arrow_table["vector"].to_pylist()
        vectors = np.array(vectors_list, dtype=np.float32)
        
        logger.info(f"Exported shape: {vectors.shape}")
        
        np.save(output_path, vectors)
        logger.info(f"Saved embeddings to {output_path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")

if __name__ == "__main__":
    export_embeddings()
