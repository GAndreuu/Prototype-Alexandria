"""
Mass Arxiv Download - Electromagnetic Biology & Growth Patterns
Downloads ~2000 papers to data/library/arxiv/
Then use mass_ingest.py to index them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import arxiv
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def download_papers():
    """Download ~2000 papers on targeted topics"""
    
    download_dir = Path("data/library/arxiv")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    
    queries = [
        # EM Biology (80 each)
        "electromagnetic fields biological effects",
        "radiofrequency radiation bees insects",
        "cell tower wildlife electromagnetic",
        "5G radiation biological impact",
        
        # Magnetoreception (80 each)
        "magnetoreception birds navigation",
        "cryptochrome magnetic field sensing",
        "radical pair mechanism biology",
        "geomagnetic navigation animals",
        
        # Fungal Networks (80 each)
        "mycelial network topology",
        "fungal growth patterns",
        "mycorrhizal networks plant communication",
        "slime mold Physarum computation",
        
        # Energy & Biosfera (80 each)
        "energy flow food webs ecosystems",
        "trophic cascade dynamics",
        "bioelectricity signaling plants",
        "ecosystem thermodynamics entropy",
        
        # Growth Algorithms (80 each)
        "fractal branching patterns biology",
        "diffusion limited aggregation",
        "Turing patterns reaction-diffusion",
        "self-organization biological systems",
        
        # Applied Physics/Math (80 each)
        "Maxwell equations biological tissues",
        "electromagnetic wave propagation tissue",
        "network theory ecology",
        "complex systems biology mathematics",
        "population dynamics radiation"
    ]
    
    total = 0
    skipped = 0
    
    logger.info("="*80)
    logger.info(f"MASS ARXIV DOWNLOAD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info(f"Target: {len(queries)} queries × 80 papers = {len(queries)*80} papers\n")
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n[{i}/{len(queries)}] Query: '{query}'")
        logger.info("-"*80)
        
        search = arxiv.Search(
            query=query,
            max_results=80,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        downloaded_this_query = 0
        
        try:
            for result in client.results(search):
                # Safe filename
                safe_title = "".join([c for c in result.title if c.isalnum() or c in (' ', '-', '_')]).strip()
                safe_title = safe_title.replace(' ', '_')[:100]
                filename = f"{safe_title}.pdf"
                filepath = download_dir / filename
                
                if filepath.exists():
                    skipped += 1
                    continue
                
                try:
                    result.download_pdf(dirpath=str(download_dir), filename=filename)
                    total += 1
                    downloaded_this_query += 1
                    
                    if downloaded_this_query % 10 == 0:
                        logger.info(f"  Downloaded: {downloaded_this_query}/80")
                        
                except Exception as e:
                    logger.error(f"  Error downloading '{result.title}': {e}")
                    continue
            
            logger.info(f"  ✅ Query complete: {downloaded_this_query} new papers")
            
        except Exception as e:
            logger.error(f"  ❌ Query failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"New papers downloaded: {total}")
    logger.info(f"Already existed (skipped): {skipped}")
    logger.info(f"Total papers in {download_dir}: {len(list(download_dir.glob('*.pdf')))}")
    logger.info(f"\nNext step: python scripts/mass_ingest.py")

if __name__ == "__main__":
    download_papers()
