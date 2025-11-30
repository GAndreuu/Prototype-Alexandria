"""
Visualizador de Topologia Semântica (3D)
Gera um gráfico 3D dos vetores armazenados no LanceDB para visualizar clusters de conhecimento.

Uso:
    python scripts/visualize_topology.py

Autor: Prototype Alexandria Team
Data: 2025-11-28
"""

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from core.storage import LanceDBStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_domain(filename):
    """Heurística simples para detectar domínio pelo nome do arquivo."""
    fn = filename.lower()
    if any(x in fn for x in ['quantum', 'physics', 'gravity', 'thermodynamics', 'particle', 'matter']):
        return 'Physics'
    if any(x in fn for x in ['bio', 'gene', 'cell', 'protein', 'neuro', 'brain', 'crispr']):
        return 'Biology'
    if any(x in fn for x in ['learning', 'neural', 'network', 'gpt', 'ai', 'intelligence', 'adversarial']):
        return 'AI/ML'
    if any(x in fn for x in ['math', 'algebra', 'topology', 'equation', 'theorem', 'stochastic']):
        return 'Math'
    if any(x in fn for x in ['compress', 'coding', 'information', 'shannon']):
        return 'InfoTheory'
    return 'Other'

def visualize():
    logger.info("🎨 Iniciando visualização da topologia (Modo Avançado)...")
    
    try:
        storage = LanceDBStorage()
        count = storage.count()
        
        if count < 10:
            logger.warning("Poucos dados.")
            return

        # Carregar dados (limitar a 5000 para t-SNE não demorar uma eternidade)
        limit = 5000 
        logger.info(f"Carregando amostra de {limit} vetores...")
        
        table = storage.table
        df = table.to_pandas()
        
        if len(df) > limit:
            df = df.sample(limit)
            
        vectors = np.stack(df['vector'].values)
        sources = df['source'].values
        
        # Detectar Domínios
        domains = [get_domain(s) for s in sources]
        unique_domains = list(set(domains))
        logger.info(f"Domínios detectados: {unique_domains}")
        
        # Mapear cores
        domain_colors = {
            'Physics': '#e74c3c',    # Red
            'Biology': '#2ecc71',    # Green
            'AI/ML': '#3498db',      # Blue
            'Math': '#9b59b6',       # Purple
            'InfoTheory': '#f1c40f', # Yellow
            'Other': '#95a5a6'       # Gray
        }
        colors = [domain_colors.get(d, '#95a5a6') for d in domains]

        # t-SNE (Melhor para visualizar clusters/atração)
        from sklearn.manifold import TSNE
        logger.info(f"Executando t-SNE (Isso pode demorar um pouco)...")
        tsne = TSNE(n_components=3, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        vectors_3d = tsne.fit_transform(vectors)
        
        # Plotar
        fig = plt.figure(figsize=(16, 12)) # Maior
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotar cada domínio separadamente para a legenda funcionar
        for domain in unique_domains:
            mask = [d == domain for d in domains]
            if not any(mask): continue
            
            d_vecs = vectors_3d[mask]
            d_color = domain_colors.get(domain, '#95a5a6')
            
            ax.scatter(
                d_vecs[:, 0], 
                d_vecs[:, 1], 
                d_vecs[:, 2], 
                c=d_color, 
                label=domain,
                alpha=0.6,
                s=15, # Pontos menores
                edgecolors='none'
            )
        
        ax.set_title(f"Universo Semântico (t-SNE) - {len(df)} Amostras", fontsize=16, color='white')
        
        # Estilo Dark/Cyberpunk para combinar com o app
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.grid(False) 
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Eixos brancos
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        ax.legend(loc='upper right', fontsize='large')
        
        output_file = "topology_viz_3d.png"
        plt.savefig(output_file, dpi=150, facecolor='#0e1117')
        logger.info(f"✅ Gráfico salvo em: {output_file}")
            
    except Exception as e:
        logger.error(f"Erro na visualização: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize()
