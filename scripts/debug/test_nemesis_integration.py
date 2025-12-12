
import sys
import numpy as np
import lancedb
import time
from typing import List

# Add path
sys.path.append('c:\\Users\\G\\Desktop\\Alexandria')
sys.path.append('c:\\Users\\G\\Desktop\\Alexandria\\core\\learning')

from core.learning.integration_layer import (
    IntegrationConfig, 
    AlexandriaIntegratedPipeline, 
    MultiAgentOrchestrator, 
    SystemProfile
)

def test_integration_on_real_data():
    print("="*60)
    print("🧬 NEMESIS INTEGRATION TEST (REAL DATA)")
    print("="*60)

    # 1. Connect to Data Source (LanceDB)
    db_path = 'c:\\Users\\G\\Desktop\\Alexandria\\data\\lancedb_store'
    print(f"\n[1] Connecting to Vector Store: {db_path}")
    
    try:
        db = lancedb.connect(db_path)
        # Assuming the table name might be 'semantic_memory' or similar based on previous checks
        # falling back to first available table if strict name fails
        tables = db.table_names()
        if not tables:
            print("❌ No tables found in LanceDB. Cannot run test on real data.")
            return
            
        table_name = tables[0]
        tbl = db.open_table(table_name)
        print(f"   ✅ Connected to table '{table_name}' ({len(tbl)} rows)")
        
        # Fetch a small sample of real data
        # We want the vector/embedding and preferably some text context if available
        sample_df = tbl.search().limit(3).to_pandas()
        samples = []
        
        for idx, row in sample_df.iterrows():
            # LanceDB usually stores vector in 'vector' column
            vec = row.get('vector')
            text = row.get('text', 'No text content')
            if vec is not None:
                samples.append((text, vec))
                
        print(f"   ✅ Loaded {len(samples)} sample chunks for testing")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # Create dummy data for fallback test
        print("   ⚠️ Switching to DUMMY data for pipeline test")
        samples = [("Dummy content A", np.random.randn(384)), ("Dummy content B", np.random.randn(384))]

    # 2. Initialize Nemesis Pipeline (LITE Mode)
    print("\n[2] Initializing Cognitive Nemesis (Profile: LITE)...")
    config = IntegrationConfig(profile=SystemProfile.LITE)
    # Mocking embedding model since we already have vectors (or fallback)
    # The pipeline usually expects raw text and does embedding internally, 
    # but we will perform a 'bypass' or inject vectors directly if possible.
    # Actually, the Orchestrator's `run_cycle` takes text. 
    # If we want to use the PRE-COMPUTED vectors, we need to bypass the embedding step in the pipeline
    # OR we just let the pipeline re-embed (but we assume we don't have the model loaded to save RAM).
    
    # Strategy: Pass the vector directly to the orchestrator methods if supported, 
    # or subclass/mock the embedding model to return our pre-computed vector.
    
    class MockEmbeddingModel:
        def encode(self, text):
            # Return a random vector or the one we grabbed if we could map it (hard with just text input)
            # For this integration test, valid 384D noise is enough to test the *Agents'* logic.
            return np.random.randn(384) 

    pipeline = AlexandriaIntegratedPipeline(
        config=config,
        embedding_model=MockEmbeddingModel() # Lite mode: avoid loading 500MB model for this test
    )
    
    # Initialize Orchestrator
    orchestrator = MultiAgentOrchestrator(pipeline)
    print("   ✅ Orchestrator & Agents (Scout, Judge, Weaver) Active")

    # 3. Process Samples
    print("\n[3] Running Agent Cycles...")
    
    for i, (text, vec) in enumerate(samples):
        print(f"\n   💠 Processing Item {i+1}: '{text[:60]}...'")
        
        start_t = time.time()
        # Note: We are passing text. The MockEmbedding above will generate a vector.
        # In a real run, we would load the SentenceTransformer.
        # This tests the FLOW: Perception -> Agents -> Mycelial Graph.
        
        results = orchestrator.run_cycle(text)
        
        # Analyze Results
        print(f"      ⏱️ Time: {time.time()-start_t:.2f}s")
        
        # Check Agent Activity
        for agent_name in ['scout', 'judge', 'weaver']:
            actions = results.get(agent_name, [])
            if actions:
                print(f"      🤖 {agent_name.upper()}: {len(actions)} actions")
                for act in actions[:1]: # Show first action
                    print(f"         - {act.get('explanation', str(act))}")
            else:
                print(f"      😴 {agent_name.upper()}: Silent")
                
        # Check Shared Memory Impact
        # We can inspect the Mycelial Graph via the pipeline
        graph_size = len(pipeline.mycelial.graph) if pipeline.mycelial else 0
        print(f"      🧠 Graph Nodes: {graph_size}")

    print("\n" + "="*60)
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_integration_on_real_data()

