"""
Script de Migração: JSON -> LanceDB
Migra o índice legado (knowledge.sfs) para o novo armazenamento LanceDB.

Uso:
    python scripts/migrate_to_lancedb.py

Autor: Prototype Alexandria Team
Data: 2025-11-28
"""

import sys
import os
import json
import logging
from pathlib import Path

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from core.storage import LanceDBStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    json_path = settings.INDEX_FILE
    
    if not os.path.exists(json_path):
        logger.warning(f"Arquivo de índice antigo não encontrado: {json_path}")
        return

    logger.info(f"Lendo índice legado: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        storage = LanceDBStorage()
        
        total_chunks = 0
        
        for file_id, metadata in data.items():
            ids = []
            vectors = []
            contents = []
            sources = []
            modalities = []
            extra_metadatas = []
            
            file_path = metadata.get('file_path', 'unknown')
            modality = metadata.get('modalidade', 'TEXTUAL')
            
            for chunk in metadata.get('indexed_chunks', []):
                ids.append(chunk['chunk_id'])
                vectors.append(chunk['vector'])
                contents.append(chunk['content'])
                sources.append(file_path)
                modalities.append(modality)
                
                extra = {
                    'file_type': metadata.get('file_type', ''),
                    'file_size': metadata.get('file_size', 0),
                    'chunk_size': chunk.get('chunk_size', 0)
                }
                extra_metadatas.append(extra)
            
            if ids:
                storage.add(ids, vectors, contents, sources, modalities, extra_metadatas)
                total_chunks += len(ids)
                
        logger.info(f"✅ Migração concluída! {total_chunks} chunks transferidos para LanceDB.")
        
        # Renomear arquivo antigo para backup
        backup_path = json_path + ".bak"
        os.rename(json_path, backup_path)
        logger.info(f"Arquivo antigo renomeado para: {backup_path}")
        
    except Exception as e:
        logger.error(f"Erro na migração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate()
