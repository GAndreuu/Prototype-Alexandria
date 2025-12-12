import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.semantic_memory import SemanticFileSystem
from core.topology.topology_engine import TopologyEngine
from core.reasoning.mycelial_reasoning import MycelialVQVAE

LOG_FILE = "query_results_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

import pypdf
import re

def normalize(text):
    """Normalize whitespace to single spaces"""
    return " ".join(text.split())

def recover_full_text(file_path, truncated_content):
    """Recover the original 1000-char chunk from PDF using the truncated snippet"""
    try:
        # Prepare search snippet: remove ellipsis, normalize
        clean_snippet = truncated_content.replace("...", "")
        search_snippet = normalize(clean_snippet)[:80]  # Use 80 chars for more precise matching
        
        if len(search_snippet) < 10:
            return f"[Snippet too short to search] {truncated_content}"
        
        # Extract full text from PDF
        reader = pypdf.PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
        
        # Recreate chunks using the same logic as SemanticFileSystem (paragraph-based, max 1000 chars)
        paragraphs = full_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= 1000:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Find the chunk that contains the search snippet
        search_snippet_norm = normalize(search_snippet)
        for chunk in chunks:
            chunk_norm = normalize(chunk)
            if search_snippet_norm in chunk_norm:
                return chunk  # Return the full original chunk (not normalized, preserves formatting)
        
        # Fallback: search in normalized text
        full_text_norm = normalize(full_text)
        if search_snippet_norm in full_text_norm:
            # If we can't find it in chunks, return a 1000-char window
            start_idx = full_text_norm.find(search_snippet_norm)
            end_idx = min(len(full_text_norm), start_idx + 1000)
            return full_text_norm[start_idx:end_idx]
            
        return f"[Could not recover chunk. Snippet not found in PDF.] Original: {truncated_content}"
        
    except Exception as e:
        return f"[Error reading PDF: {e}]. Snippet: {truncated_content}"

def run_query():
    # Clear log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    log("🧠 Initializing Systems...")
    
    query = "vector quantization power law neural criticality"
    
    try:
        topology = TopologyEngine()
        # Check for fallback mode - we do NOT want random vectors for this critical analysis
        if getattr(topology, 'use_fallback', False):
            log("❌ CRITICAL: TopologyEngine is using fallback (random vectors). Aborting to prevent garbage data.")
            log("   Tip: Stop other heavy processes (ingestion) and try again.")
            return
            
        # DEBUG: Check model and vector
        log(f"🔍 Topology Model: {topology.model_name} on {topology.device}")
        test_vec = topology.encode([query])[0]
        log(f"🔍 Query Vector Sample (first 5): {test_vec[:5]}")
        log(f"🔍 Query Vector Norm: {np.linalg.norm(test_vec)}")
        with open("vector_debug_query.txt", "w") as f:
            f.write(str(test_vec.tolist()))
            
    except Exception as e:
        log(f"❌ Error initializing topology: {e}")
        return
        
    try:
        # WORKAROUND: Use direct LanceDB search because SemanticFileSystem.retrieve is behaving inconsistently
        # (likely due to vector normalization/passing issues in the wrapper)
        import lancedb
        import os
        from config import settings
        
        db_path = os.path.join(settings.DATA_DIR, "lancedb_store")
        db = lancedb.connect(db_path)
        tbl = db.open_table("semantic_memory")
        
        # Generate and normalize query vector
        query_vec = topology.encode([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        log(f"🔍 Executing Direct LanceDB Search (Limit 20)...")
        results = tbl.search(query_vec).limit(20).to_list()
        
        # Format results to match expected structure if needed (LanceDB returns dicts, so should be fine)
        # We need to ensure 'relevance' is calculated if not present
        for r in results:
            dist = r.get('_distance', 1.0)
            r['relevance'] = max(0, 1 - (dist / 2))
            
    except Exception as e:
        log(f"❌ Error during retrieval: {e}")
        return
    
    log(f"✅ Found {len(results)} results (showing top {min(len(results), 20)}).\n")
    
    # Export for synthesis
    with open("synthesis_data.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write("="*80 + "\n\n")
        
        for i, res in enumerate(results):
            source_name = Path(res['source']).name
            log(f"Processing [{i+1}/20]: {source_name}")
            
            # Recover full text
            full_content = recover_full_text(res['source'], res['content'])
            
            f.write(f"Source: {source_name}\n")
            f.write(f"Relevance: {res['relevance']:.4f}\n")
            f.write("-" * 40 + "\n")
            f.write(full_content)
            f.write("\n" + "="*80 + "\n\n")
            
    log(f"📄 Full text chunks (Top 20) saved to 'synthesis_data.txt'")

    for i, res in enumerate(results):
        log(f"[{i+1}] {Path(res['source']).name}")
        log(f"    Relevance: {res['relevance']:.4f}")
        # ... rest of logging ...
        log(f"    Snippet: {res['content'][:100]}...")
        
        # Get vector
        vector = res.get('vector')
        if vector:
            # Convert to tensor
            x = torch.tensor(vector, dtype=torch.float32)
            
            # Run Mycelial Reasoning
            try:
                out = mycelial_system.full_pipeline(x, reason=True)
                
                orig_idx = out['original_indices'].tolist()[0]
                reasoned_idx = out['reasoned_indices'].tolist()[0] # Convert tensor to list
                
                log(f"    🧩 VQ Codes: {orig_idx}")
                
                if orig_idx != reasoned_idx:
                    log(f"    💡 Reasoning Expanded to: {reasoned_idx}")
                    # Check what changed
                    changes = [k for k in range(4) if orig_idx[k] != reasoned_idx[k]]
                    log(f"       (Changed heads: {changes})")
                else:
                    log(f"    ⚪ No expansion triggered (Stable Concept).")
            except Exception as e:
                log(f"    ⚠️ Reasoning failed: {e}")
        else:
            log("    ⚠️ No vector found.")
        log("-" * 60)

if __name__ == "__main__":
    run_query()
