"""
Script de Ingestão de Livros (Library)
Ingere PDFs da pasta data/library diretamente para o LanceDB.
"""

import sys
import os
import logging
import pypdf
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from core.storage import LanceDBStorage
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BookIngest")

def extract_text_from_pdf(file_path: Path) -> str:
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Erro ao ler PDF {file_path.name}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Limpar whitespace excessivo
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def ingest_books():
    logger.info("📚 Iniciando Ingestão de Livros...")
    
    # Setup
    encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
    storage = LanceDBStorage()
    
    library_path = Path(settings.DATA_DIR) / "uploads"
    if not library_path.exists():
        logger.error(f"Pasta uploads não encontrada: {library_path}")
        return

    files = list(library_path.glob("*.pdf"))
    logger.info(f"Encontrados {len(files)} livros em {library_path}")
    
    total_chunks = 0
    
    for file_path in tqdm(files, desc="Processando Livros"):
        logger.info(f"📖 Lendo: {file_path.name}")
        
        # 1. Extrair Texto
        content = extract_text_from_pdf(file_path)
        if not content:
            continue
            
        # 2. Chunking
        chunks = chunk_text(content, chunk_size=settings.CHUNK_SIZE)
        if not chunks:
            continue
            
        logger.info(f"   - Gerados {len(chunks)} chunks (tamanho ~{settings.CHUNK_SIZE})")
        
        # 3. Encoding
        vectors = encoder.encode(chunks, show_progress_bar=False)
        
        # 4. Preparar Batch para LanceDB
        ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        # Converter numpy vectors para list
        vectors_list = vectors.tolist()
        
        sources = [str(file_path)] * len(chunks)
        modalities = ["TEXTUAL"] * len(chunks)
        
        extra_metadatas = []
        for chunk in chunks:
            extra_metadatas.append({
                'file_type': '.pdf',
                'file_size': len(content),
                'chunk_size': len(chunk),
                'source_type': 'book'
            })
            
        # 5. Salvar
        storage.add(ids, vectors_list, chunks, sources, modalities, extra_metadatas)
        total_chunks += len(chunks)
        
    logger.info(f"✅ Ingestão de Livros Concluída! Total de chunks: {total_chunks}")

if __name__ == "__main__":
    ingest_books()
