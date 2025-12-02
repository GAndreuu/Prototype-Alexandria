"""
Custom Arxiv Harvester - Electromagnetic Biology & Growth Patterns
Uses ArxivHarvester.harvest() to download and auto-ingest papers
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.harvester import ArxivHarvester
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def harvest_electromagnetic_biology():
    """Download papers on EM biology, magnetorece,ption, fungal networks, growth patterns"""
    
    harvester = ArxivHarvester()
    
    # All queries combined
    queries = [
        # Electromagnetic Biology
        "electromagnetic fields biological effects",
        "radiofrequency radiation insects pollinators",
        "cell tower wildlife impact",
        "RF exposure bees honeybees",
        
        # Magnetoreception 
        "magnetoreception birds navigation",
        "cryptochrome magnetic sensing",
        "radical pair mechanism magnetosensitivity",
        "geomagnetic field animal navigation",
        
        # Fungal Networks
        "mycelial network topology graph theory",
        "fungal growth patterns algorithms",
        "mycorrhizal networks communication",
        "slime mold computational intelligence",
        
        # Biosphere Energy
        "energy flow ecosystems food webs",
        "trophic cascade thermodynamics",
        "bioelectricity plant signaling",
        
        # Growth Algorithms & Patterns
        "biological branching fractal mathematics",
        "diffusion limited aggregation biology",
        "Turing patterns reaction-diffusion",
        "self-organizing biological systems",
        
        # Applied Math/Physics
        "Maxwell equations biological tissues",
        "network theory ecology",
        "population dynamics electromagnetic radiation"
    ]
    
    logger.info(f"Starting harvest of {len(queries)} queries")
    logger.info(f"Target: 80 papers per query = ~{len(queries)*80} papers total\n")
    
    # Harvest (auto-downloads and ingests into LanceDB)
    harvester.harvest(queries, max_per_query=80, ingest=True)
    
    logger.info("\n" + "="*80)
    logger.info("HARVEST COMPLETE")
    logger.info("="*80)
    logger.info("Papers are now indexed in Alexandria!")
    logger.info("\nYou can now use:")
    logger.info("  - Semantic search for these topics")
    logger.info("  - Semantic collider between domains")
    logger.info("  - Causal analysis of connections")

if __name__ == "__main__":
    harvest_electromagnetic_biology()
