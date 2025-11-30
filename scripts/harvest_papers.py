"""
Script de Colheita de Papers
Executa o ArxivHarvester para buscar e ingerir papers sobre compressão e tecnologia.

Uso:
    python scripts/harvest_papers.py

Autor: Prototype Alexandria Team
Data: 2025-11-28
"""

import sys
import logging
from pathlib import Path

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from core.harvester import ArxivHarvester

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Tópicos de pesquisa massiva (Multi-domínio)
    topics = [
        # Machine Learning & AI
        "deep learning", "reinforcement learning", "transformers nlp", "computer vision", "generative adversarial networks",
        "large language models", "graph neural networks", "bayesian optimization", "explainable ai", "neuromorphic computing",
        
        # Physics
        "quantum physics", "astrophysics", "condensed matter physics", "particle physics", "general relativity",
        "string theory", "thermodynamics", "fluid dynamics", "quantum computing", "dark matter",
        
        # Mathematics
        "topology", "number theory", "algebraic geometry", "category theory", "chaos theory",
        "differential equations", "graph theory", "stochastic processes", "game theory", "cryptography",
        
        # Biology & Life Sciences
        "molecular biology", "genetics", "neuroscience", "evolutionary biology", "CRISPR",
        "bioinformatics", "synthetic biology", "immunology", "protein folding", "ecology",
        
        # Interdisciplinary & Others
        "complex systems", "nanotechnology", "cognitive science", "robotics", "materials science",
        "climate change modeling", "network science", "cybernetics", "information theory"
    ]
    
    print("🚀 Iniciando Protocolo de Colheita Massiva de Conhecimento...")
    print(f"📚 Tópicos Alvo: {len(topics)}")
    
    harvester = ArxivHarvester()
    
    # Colher 100 papers de cada tópico (Total ~4500 papers)
    # Nota: O ArxivHarvester já deve ter rate limiting interno.
    harvester.harvest(topics, max_per_query=100, ingest=True)
    
    print("\n✅ Missão Cumprida. O conhecimento foi assimilado.")

if __name__ == "__main__":
    main()
