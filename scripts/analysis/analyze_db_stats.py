"""
Script de Análise de Topologia
Responde perguntas sobre a distribuição dos dados no LanceDB.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.memory.storage import LanceDBStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze():
    output_path = "analysis_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("🔍 Iniciando Análise Forense dos Dados...")
        
        storage = LanceDBStorage()
        table = storage.table
        df = table.to_pandas()
        
        total_chunks = len(df)
        log(f"\n📊 Estatísticas Gerais:")
        log(f"   - Total de Chunks: {total_chunks}")
        
        # 1. Quantos Papers?
        unique_sources = df['source'].unique()
        log(f"   - Total de Fontes Únicas (Papers/Livros): {len(unique_sources)}")
        
        # Listar algumas fontes para ver diversidade
        log(f"\n📚 Amostra de Fontes:")
        for source in unique_sources[:10]:
            log(f"   - {Path(source).name}")
            
        # 2. Distribuição de Modalidade
        log(f"\n🎨 Distribuição de Modalidade:")
        modality_counts = df['modality'].value_counts()
        log(str(modality_counts))
        
        # 3. Identificar Outlier
        # Vamos rodar o mesmo PCA para achar o ponto mais distante
        vectors = np.stack(df['vector'].values)
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)
        
        # Calcular distância do centro (0,0,0)
        distances = np.linalg.norm(vectors_3d, axis=1)
        outlier_idx = np.argmax(distances)
        outlier = df.iloc[outlier_idx]
        
        log(f"\n☄️ Análise do Outlier (Ponto mais distante do centro):")
        log(f"   - ID: {outlier['id']}")
        log(f"   - Fonte: {Path(outlier['source']).name}")
        log(f"   - Modalidade: {outlier['modality']}")
        # Limpar caracteres nulos ou problemáticos do conteúdo
        content_preview = outlier['content'][:200].replace('\x00', '')
        log(f"   - Conteúdo (início): {content_preview}...")
        
        # Verificar se há imagens
        visuals = df[df['modality'] == 'VISUAL']
        if len(visuals) > 0:
            log(f"\n🖼️ Sobre as Imagens ({len(visuals)} encontradas):")
            log(f"   - Exemplo: {visuals.iloc[0]['source']}")
        else:
            log(f"\n⚠️ Nenhuma imagem encontrada no dataset atual.")
            
    print(f"Relatório salvo em {output_path}")

if __name__ == "__main__":
    analyze()
