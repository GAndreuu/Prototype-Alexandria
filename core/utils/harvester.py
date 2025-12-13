"""
Prototype Alexandria - Paper Harvester
Módulo de aquisição automática de conhecimento científico via Arxiv.

Foca em tópicos de compressão, matemática e tecnologia para alimentar
o ciclo de aprendizado do sistema.

Autor: Prototype Alexandria Team
Data: 2025-11-28
"""

import os
import logging
import arxiv
from pathlib import Path
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger(__name__)

class ArxivHarvester:
    """
    Colheitadeira de Papers do Arxiv.
    Busca, baixa e ingere papers automaticamente.
    """
    
    def __init__(self, download_dir: str = None):
        self.download_dir = download_dir or os.path.join(settings.DATA_DIR, "library", "arxiv")
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Cliente Arxiv
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )
        
        logger.info(f"🌾 ArxivHarvester inicializado. Destino: {self.download_dir}")

    def search_papers(self, query: str, max_results: int = 10) -> List[Any]:
        """Busca papers no Arxiv."""
        logger.info(f"🔍 Buscando '{query}' no Arxiv (max: {max_results})...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        try:
            for result in self.client.results(search):
                results.append(result)
            logger.info(f"✅ Encontrados {len(results)} papers.")
            return results
        except Exception as e:
            logger.error(f"❌ Erro na busca Arxiv: {e}")
            return []

    def download_paper(self, paper: Any) -> str:
        """
        Baixa o PDF de um paper.
        Retorna o caminho do arquivo salvo.
        """
        try:
            # Sanitizar título para nome de arquivo
            safe_title = "".join([c for c in paper.title if c.isalnum() or c in (' ', '-', '_')]).strip()
            safe_title = safe_title.replace(' ', '_')[:100] # Limitar tamanho
            filename = f"{safe_title}.pdf"
            
            # Verificar se já existe
            file_path = os.path.join(self.download_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"⏭️ Paper já existe: {filename}")
                return file_path
            
            # Baixar
            logger.info(f"⬇️ Baixando: {paper.title}")
            paper.download_pdf(dirpath=self.download_dir, filename=filename)
            logger.info(f"✅ Download concluído: {filename}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar paper {paper.title}: {e}")
            return None

    def harvest(self, queries: List[str], max_per_query: int = 5, ingest: bool = True):
        """
        Executa o fluxo completo: Busca -> Download -> Ingestão.
        """
        # Importar aqui para evitar ciclo se não for usar
        from core.topology.topology_engine import create_topology_engine
        from core.memory.semantic_memory import SemanticFileSystem
        
        sfs = None
        if ingest:
            # Inicializar sistema de memória
            topology = create_topology_engine()
            # Tentar carregar encoder real se possível, senão fallback
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                encoder = None
                
            sfs = SemanticFileSystem(topology, engine_encoder=encoder)
        
        total_ingested = 0
        
        for query in queries:
            papers = self.search_papers(query, max_results=max_per_query)
            
            for paper in papers:
                pdf_path = self.download_paper(paper)
                
                if pdf_path and ingest and sfs:
                    # Ingerir no LanceDB
                    logger.info(f"🧠 Ingerindo: {Path(pdf_path).name}")
                    chunks = sfs.index_file(pdf_path, doc_type="SCI")
                    if chunks > 0:
                        total_ingested += 1
        
        if ingest:
            logger.info(f"🎉 Colheita finalizada! Total de documentos ingeridos: {total_ingested}")
