#!/usr/bin/env python3
"""
Cyclic Arxiv Harvest - Alexandria Taxonomy
==========================================

Downloads papers in a continuous cycle based on the Alexandria Taxonomy:
- Mathematics
- Physics
- Machine Learning
- Neuroscience

Target: ~900 papers per cycle (approx 50 per subtopic).
"""
import sys
from pathlib import Path
import time
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import arxiv
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("harvest_cycle.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Force utf-8 for stdout if possible, or just rely on handler
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

TAXONOMY = {
    "MATEMÁTICA": [
        "Information Theory",
        "Topology",
        "Linear Algebra",
        "Graph Theory",
        "Statistics",
        "Category Theory"
    ],
    "FÍSICA": [
        "Statistical Mechanics",
        "Criticality",
        "Complexity Theory",
        "Emergence",
        "Thermodynamics",
        "Renormalization Group"
    ],
    "MACHINE LEARNING": [
        "Vector Quantization VQ-VAE",
        "Representation Learning",
        "Sparse Coding Dictionary Learning",
        "Transformers Self-Attention",
        "Contrastive Learning SimCLR CLIP",
        "Neural Memory Networks",
        "Retrieval Augmented Generation RAG",
        "Graph Neural Networks GNN",
        "Mixture of Experts MoE",
        "Variational Autoencoders Disentanglement"
    ],
    "NEUROCIÊNCIA": [
        "Hebbian Learning",
        "Associative Memory",
        "Semantic Cognition",
        "Memory Consolidation",
        "Predictive Coding"
    ],
    "IA SIMBÓLICA": [
        "Knowledge Graphs",
        "Semantic Networks",
        "Frame Systems Minsky",
        "Analogical Reasoning",
        "Abductive Reasoning",
        "Neuro-symbolic AI"
    ]
}

def get_safe_filename(title: str) -> str:
    """Creates a safe filename from title."""
    safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()
    safe_title = safe_title.replace(' ', '_')[:100]
    return f"{safe_title}.pdf"

def run_harvest_cycle(papers_per_subtopic: int = 50):
    """Runs one full harvest cycle."""
    
    download_dir = Path("data/library/arxiv")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    
    total_downloaded = 0
    cycle_start = datetime.now()
    
    logger.info("="*60)
    logger.info(f"INICIANDO CICLO DE COLHEITA - {cycle_start}")
    logger.info(f"Meta: {papers_per_subtopic} papers por sub-tópico")
    logger.info("="*60)
    
    for category, topics in TAXONOMY.items():
        logger.info(f"\n>>> CATEGORIA: {category}")
        
        for topic in topics:
            logger.info(f"   Tópico: {topic}")
            
            # Construct query
            query = f'all:"{topic}"'
            
            # Search deeper (up to 3000) to find new papers
            search = arxiv.Search(
                query=query,
                max_results=3000, 
                sort_by=arxiv.SortCriterion.SubmittedDate 
            )
            
            topic_downloaded = 0
            checked_count = 0
            
            try:
                # Iterate through results until we get enough NEW papers or run out
                for result in client.results(search):
                    checked_count += 1
                    
                    # Stop if we have enough new papers for this topic
                    if topic_downloaded >= papers_per_subtopic:
                        break
                        
                    filename = get_safe_filename(result.title)
                    filepath = download_dir / filename
                    
                    if filepath.exists():
                        continue
                    
                    try:
                        result.download_pdf(dirpath=str(download_dir), filename=filename)
                        topic_downloaded += 1
                        total_downloaded += 1
                        
                        if topic_downloaded % 10 == 0:
                            logger.info(f"      Baixados: {topic_downloaded}/{papers_per_subtopic} (Verificados: {checked_count})")
                            
                    except Exception as e:
                        logger.error(f"      Erro ao baixar '{result.title}': {e}")
                        continue
                
                logger.info(f"   ✅ Tópico concluído: {topic_downloaded} novos papers (Verificados: {checked_count})")
                
            except Exception as e:
                logger.error(f"   ❌ Erro na busca por '{topic}': {e}")
            
            # Small pause between topics to be nice to Arxiv
            time.sleep(2)
            
    logger.info("\n" + "="*60)
    logger.info(f"CICLO CONCLUÍDO - Total: {total_downloaded} papers")
    logger.info("="*60)
    return total_downloaded

def main():
    cycle_count = 1
    
    while True:
        logger.info(f"\n\n🔁 INICIANDO CICLO #{cycle_count}")
        
        # 900 papers total / 19 topics ~= 48 papers per topic
        # Rounding to 50
        count = run_harvest_cycle(papers_per_subtopic=50)
        
        logger.info(f"Ciclo #{cycle_count} finalizado com {count} downloads.")
        logger.info("Aguardando 60 segundos para reiniciar...")
        time.sleep(60)
        
        cycle_count += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n🛑 Colheita interrompida pelo usuário.")
