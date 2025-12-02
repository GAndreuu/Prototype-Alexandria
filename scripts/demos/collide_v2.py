"""
Semantic Collider V2 - Research Grade
Generates rigorous scientific analysis with citations, not philosophy.
"""
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from core.memory.storage import LanceDBStorage
from config import settings

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Collider")

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith(("GOOGLE_API_KEY=", "GEMINI_API_KEY=")):
                        api_key = line.strip().split("=")[1]
                        break
        except:
            pass
            
    if not api_key:
        logger.warning("⚠️  API KEY not found. Analysis will be limited.")
        return None
        
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(settings.GEMINI_MODEL)

def collide(source_query: str, target_query: str, min_connections=10, output_path="collision_report.txt"):
    """
    Generate research-grade collision report.
    
    Args:
        source_query: Source domain
        target_query: Target domain  
        min_connections: Minimum connections to analyze
        output_path: Output file path
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        storage = LanceDBStorage()
        table = storage.table
        
        log("="*80)
        log(f"SEMANTIC COLLISION ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("="*80)
        log(f"\nDOMAIN A: {source_query}")
        log(f"DOMAIN B: {target_query}\n")
        
        # 1. Collect source documents
        source_filter = f"source LIKE '%{source_query}%'"
        source_results = table.search().where(source_filter).limit(50).to_list()
        
        if not source_results:
            log(f"ERROR: No documents found matching '{source_query}'")
            return

        log(f"Found {len(source_results)} source documents from Domain A")
        
        # 2. Find bridges
        collisions = []
        seen_pairs = set()

        for s_chunk in source_results:
            source_filename = Path(s_chunk['source']).name
            
            target_filter = f"source NOT LIKE '%{source_filename}%'"
            if target_query.lower() not in ["papers", "all"]:
                 target_filter += f" AND source LIKE '%{target_query}%'"
                 
            neighbors = table.search(s_chunk['vector']) \
                .where(target_filter) \
                .limit(10) \
                .to_list()
                
            for n in neighbors:
                target_filename = Path(n['source']).name
                
                if source_filename == target_filename:
                    continue
                    
                pair_id = f"{s_chunk['id']}-{n['id']}"
                if pair_id in seen_pairs:
                    continue
                seen_pairs.add(pair_id)

                dist = n['_distance']
                sim = 1 - (dist / 2)
                
                if sim > 0.35:  # Lower threshold to capture more
                    collisions.append({
                        'source_chunk': s_chunk,
                        'target_chunk': n,
                        'similarity': sim,
                        's_name': source_filename,
                        't_name': target_filename,
                        's_text': s_chunk['content'],
                        't_text': n['content']
                    })
        
        # Sort and deduplicate
        collisions.sort(key=lambda x: x['similarity'], reverse=True)
        
        unique_collisions = []
        seen_targets = set()
        for c in collisions:
            if c['t_name'] not in seen_targets:
                unique_collisions.append(c)
                seen_targets.add(c['t_name'])
            if len(unique_collisions) >= min_connections:
                break
                
        if not unique_collisions:
            log("\nNo significant semantic bridges found.\n")
            return

        log(f"Identified {len(unique_collisions)} high-confidence bridges\n")
        log("="*80)
        log("BRIDGE ANALYSIS")
        log("="*80 + "\n")
        
        # 3. Display ALL bridges with full context
        detailed_connections = []
        
        for i, c in enumerate(unique_collisions, 1):
            log(f"BRIDGE #{i} | Similarity: {c['similarity']:.3f}")
            log(f"Source: {c['s_name']}")
            log(f"Target: {c['t_name']}\n")
            
            log(f"Text from {c['s_name']}:")
            log(f'"{c['s_text'][:400]}..."\n')
            
            log(f"Text from {c['t_name']}:")
            log(f'"{c['t_text'][:400]}..."\n')
            
            log("-"*80 + "\n")
            
            detailed_connections.append({
                'index': i,
                'similarity': c['similarity'],
                'source_doc': c['s_name'],
                'target_doc': c['t_name'],
                'source_text': c['s_text'],
                'target_text': c['t_text']
            })
        
        # 4. Generate RESEARCH-GRADE analysis
        model = setup_gemini()
        if model:
            log("="*80)
            log("TECHNICAL SYNTHESIS")
            log("="*80 + "\n")
            
            # Build context WITH ALL CONNECTIONS
            context_parts = []
            for conn in detailed_connections[:10]:  # Top 10 for context window
                context_parts.append(
                    f"CONNECTION {conn['index']} (Similarity: {conn['similarity']:.3f}):\n"
                    f"From [{conn['source_doc']}]:\n{conn['source_text']}\n\n"
                    f"From [{conn['target_doc']}]:\n{conn['target_text']}\n"
                )
            
            full_context = "\n---\n".join(context_parts)
            
            prompt = f"""You are a technical research analyst specializing in cross-domain scientific synthesis.

TASK: Analyze the semantic bridges between "{source_query}" and "{target_query}" domains.

DATA: Below are the top semantic connections identified by vector similarity search:

{full_context}

REQUIRED OUTPUT FORMAT:

# Cross-Domain Analysis: {source_query} ↔ {target_query}

## 1. Direct Technical Connections
For EACH connection (minimum 5), provide:
- **Bridge #{'{'}i{'}'} ({'{'}similarity{'}'}%)**: One-sentence summary
- **Source Concept**: Extract the specific technical concept/equation/principle from source
- **Target Concept**: Extract the specific technical concept/equation/principle from target
- **Mathematical/Scientific Link**: WHY are these similar? What underlying principle connects them?

## 2. Quantitative Pattern Analysis
Identify recurring:
- Mathematical structures (equations, operators, formalisms)
- Conceptual frameworks (control theory, optimization, etc.)
- Methodological approaches

## 3. Novel Research Directions
Based ONLY on the concrete connections above, suggest:
- 3 specific, testable hypotheses
- Required experimental validation
- Expected outcomes

## 4. Citation Map
List all unique papers referenced with their core contribution to this cross-domain analysis.

CONSTRAINTS:
- NO philosophical speculation
- NO generic statements
- EVERY claim must cite a specific bridge number
- Focus on QUANTITATIVE and STRUCTURAL similarities
- If equations appear, EXTRACT and COMPARE them directly"""

            try:
                response = model.generate_content(prompt)
                log(response.text)
                log("\n" + "="*80 + "\n")
            except Exception as e:
                log(f"ERROR generating synthesis: {e}\n")
        
        # 5. Statistics
        log("COLLISION STATISTICS")
        log("="*80)
        log(f"Total bridges identified: {len(unique_collisions)}")
        log(f"Unique target papers: {len(seen_targets)}")
        log(f"Average similarity: {np.mean([c['similarity'] for c in unique_collisions]):.3f}")
        log(f"Max similarity: {max([c['similarity'] for c in unique_collisions]):.3f}")
        log(f"Min similarity: {min([c['similarity'] for c in unique_collisions]):.3f}")
        
        log(f"\nReport saved to: {Path(output_path).absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Collider V2 - Research Grade")
    parser.add_argument("--source", required=True, help="Source domain term")
    parser.add_argument("--target", default="papers", help="Target domain term")
    parser.add_argument("--min-connections", type=int, default=10, help="Minimum connections to analyze")
    
    args = parser.parse_args()
    collide(args.source, args.target, args.min_connections)
