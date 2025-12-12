"""
Prototype Alexandria - LanceDB Storage Engine
Armazenamento vetorial de alta performance em disco.

Substitui o sistema de arquivos JSON por LanceDB (formato columnar),
permitindo busca eficiente em milhões de vetores com baixo uso de RAM.

Autor: Prototype Alexandria Team
Data: 2025-11-28
"""

import logging
import os
import lancedb
import pyarrow as pa
import numpy as np
from typing import List, Dict, Any, Optional, Union
from config import settings

logger = logging.getLogger(__name__)

class LanceDBStorage:
    """
    Gerenciador de armazenamento persistente usando LanceDB.
    Mantém vetores e metadados em disco, carregando apenas o necessário.
    """
    
    def __init__(self, db_path: str = None):
        """
        Inicializa conexão com LanceDB.
        
        Args:
            db_path: Caminho para a pasta do banco (padrão: data/lancedb)
        """
        self.db_path = db_path or os.path.join(settings.DATA_DIR, "lancedb_store")
        os.makedirs(self.db_path, exist_ok=True)
        
        try:
            self.db = lancedb.connect(self.db_path)
            self.table_name = "semantic_memory"
            self._init_table()
            logger.info(f"✅ LanceDB conectado em: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar LanceDB: {e}")
            raise

    def _init_table(self):
        """Inicializa ou carrega a tabela de memória."""
        # Esquema PyArrow explícito para garantir tipos
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 384)),
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("source", pa.string()),
            pa.field("modality", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("metadata", pa.string()) # JSON string para flexibilidade extra
        ])
        
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            logger.info(f"Tabela '{self.table_name}' criada.")
        else:
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Tabela '{self.table_name}' carregada. Registros: {self.table.count_rows()}")

    def add(self, 
            ids: List[str], 
            vectors: List[List[float]], 
            contents: List[str], 
            sources: List[str], 
            modalities: List[str],
            extra_metadata: List[Dict] = None):
        """
        Adiciona itens ao banco.
        """
        try:
            data = []
            import json
            from datetime import datetime
            
            timestamp = datetime.now().isoformat()
            
            for i, vec in enumerate(vectors):
                meta_str = json.dumps(extra_metadata[i]) if extra_metadata else "{}"
                
                # Garantir que vetor é float32
                vec_np = np.array(vec, dtype=np.float32)
                
                data.append({
                    "vector": vec_np,
                    "id": ids[i],
                    "content": contents[i],
                    "source": sources[i],
                    "modality": modalities[i],
                    "timestamp": timestamp,
                    "metadata": meta_str
                })
            
            if data:
                self.table.add(data)
                logger.info(f"💾 {len(data)} itens adicionados ao LanceDB.")
                
        except Exception as e:
            logger.error(f"Erro ao adicionar dados ao LanceDB: {e}")
            raise

    def search(self, query_vector: List[float], limit: int = 10, filter_sql: str = None) -> List[Dict]:
        """
        Busca vetorial aproximada.
        
        Args:
            query_vector: Vetor de busca (384D)
            limit: Máximo de resultados
            filter_sql: Filtro SQL (ex: "modality = 'VISUAL'")
            
        Returns:
            Lista de resultados com score (distância)
        """
        try:
            query_np = np.array(query_vector, dtype=np.float32)
            
            search_job = self.table.search(query_np).limit(limit)
            
            if filter_sql:
                search_job = search_job.where(filter_sql)
                
            results = search_job.to_list()
            
            # Formatar saída para compatibilidade
            formatted = []
            import json
            
            for r in results:
                # LanceDB retorna distância (menor é melhor para L2)
                # Converter para "relevância" (0 a 1) aproximada
                # Assumindo vetores normalizados, dist L2 varia de 0 a 2
                # Cosine sim = 1 - (dist^2 / 2)
                dist = r.get('_distance', 1.0)
                relevance = max(0, 1 - (dist / 2)) # Aproximação simples
                
                formatted.append({
                    "id": r['id'],
                    "content": r['content'],
                    "source": r['source'],
                    "modality": r['modality'],
                    "relevance": float(relevance),
                    "metadata": json.loads(r['metadata']) if r['metadata'] else {},
                    "vector": r['vector'] # Opcional, pode remover se pesar
                })
                
            return formatted
            
        except Exception as e:
            logger.error(f"Erro na busca LanceDB: {e}")
            return []

    def count(self) -> int:
        return self.table.count_rows()
