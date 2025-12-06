"""
Probe Reasoning Stack
Diagnostic script to verify Topology and Mycelial Reasoning state using live data.
"""
import sys
import os
import numpy as np
import logging
from pathlib import Path
import lancedb
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.topology.topology_engine import TopologyEngine
from core.reasoning.mycelial_reasoning import MycelialReasoning
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PROBE")

def probe_system():
    print("="*80)
    print("PROBE REASONING STACK")
    print("="*80)

    # 1. DATABASE CHECK
    print("\n[1] Checking Database...")
    db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
    
    if not os.path.exists(db_path):
         print("    [FAIL] Database path not found.")
         return

    try:
        db = lancedb.connect(db_path)
        if "semantic_memory" not in db.table_names():
             print("    [FAIL] Table semantic_memory not found.")
             return

        table = db.open_table("semantic_memory")
        count = table.count_rows()
        print(f"    Chunks indexed: {count:,}")
        
        if count == 0:
            print("    [FAIL] Database empty. Cannot probe.")
            return

        # 2. VECTOR QUALITY
        print("\n[2] Checking Vector Quality...")
        sample = table.search([0.0]*384).limit(1).to_list()[0]
        vector = np.array(sample['vector'])
        norm = np.linalg.norm(vector)
        
        print(f"    Sample ID: {sample['id'][:8]}...")
        print(f"    Source: {sample.get('source', 'Unknown')}")
        print(f"    Vector Shape: {vector.shape}")
        print(f"    Vector Norm: {norm:.6f}")
        
        if norm < 0.1 or np.isnan(norm):
            print("    [FAIL] Vector appears invalid (zero or NaN)")
        else:
            print("    [OK] Vector looks valid")
            
    except Exception as e:
        print(f"    [FAIL] Database/Vector Error: {e}")
        return

    # 3. TOPOLOGY ENGINE
    print("\n[3] Checking Topology Engine...")
    try:
        topo = TopologyEngine()
        print(f"    Model: {topo.model_name}")
        print(f"    Device: {topo.device}")
        print(f"    Fallback Mode: {getattr(topo, 'use_fallback', False)}")
        
        test_text = "Neural networks are universal function approximators."
        emb = topo.encode([test_text])[0]
        print(f"    Live Encoding Norm: {np.linalg.norm(emb):.6f}")
        
    except Exception as e:
        print(f"    [FAIL] Topology Error: {e}")
        return

    # 4. MYCELIAL REASONING (VQ-VAE + Graph)
    print("\n[4] Checking Mycelial Reasoning...")
    try:
        brain = MycelialReasoning()
        
        # Stats
        print(f"    Total Observations: {brain.total_observations}")
        print(f"    Graph Nodes: {len(brain.graph)}")
        
        try:
            from core.learning.v2_learner import V2Learner
            learner = V2Learner()
            print("    [OK] V2Learner loaded")
            
            # Discretize sample vector
            codes = learner.encode(torch.tensor(vector).unsqueeze(0))
            print(f"    Vector Codes: {codes}")
            
        except ImportError:
            print("    [WARN] V2Learner not found (might be named differently)")
        except Exception as e:
            print(f"    [WARN] VQ-VAE Test Error: {e}")

        if len(brain.graph) == 0:
            print("    [WARN] Mycelial Graph is EMPTY (Expected if no training run)")
        else:
            print(f"    [OK] Mycelial Graph Active ({len(brain.graph)} connections)")

    except Exception as e:
        print(f"    [FAIL] Mycelial Error: {e}")

    print("\n" + "="*80)
    print("PROBE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    probe_system()
