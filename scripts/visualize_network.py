"""
Visualizador de Rede Semântica 3D (Interativo)
Gera um grafo 3D onde:
- Nós = Chunks (coloridos por domínio)
- Arestas = Similaridade > Threshold (0.7)
- Saída = Arquivo HTML interativo (Plotly)

Uso:
    python scripts/visualize_network.py
"""

import sys
import logging
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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

def generate_network_graph():
    logger.info("🕸️ Iniciando geração do Grafo de Rede 3D...")
    
    try:
        storage = LanceDBStorage()
        
        # Limitar nós para não travar o browser (grafos 3D são pesados)
        limit = 1000
        logger.info(f"Carregando {limit} nós...")
        
        table = storage.table
        df = table.to_pandas()
        
        if len(df) > limit:
            df = df.sample(limit, random_state=42) # Random state para consistência
            
        vectors = np.stack(df['vector'].values)
        ids = df['id'].values
        contents = df['content'].values
        sources = df['source'].values
        
        # 1. Calcular Matriz de Similaridade
        logger.info("Calculando similaridade cosseno...")
        sim_matrix = cosine_similarity(vectors)
        
        # 2. Construir Grafo NetworkX
        logger.info("Construindo grafo (Threshold > 0.7)...")
        G = nx.Graph()
        
        # Adicionar nós com metadados
        for i in range(len(df)):
            domain = get_domain(sources[i])
            G.add_node(i, id=ids[i], domain=domain, content=contents[i][:200], source=sources[i])
            
        # Adicionar arestas
        threshold = 0.7
        rows, cols = np.where(sim_matrix > threshold)
        for r, c in zip(rows, cols):
            if r < c: # Evitar duplicatas e auto-loops
                weight = sim_matrix[r, c]
                G.add_edge(r, c, weight=weight)
                
        logger.info(f"Grafo criado: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
        
        # 3. Layout 3D (Spring Layout)
        logger.info("Calculando layout 3D (pode demorar)...")
        # k controla a distância entre nós. Menor = mais agrupado.
        pos = nx.spring_layout(G, dim=3, seed=42, k=0.5, iterations=50)
        
        # 4. Preparar dados para Plotly
        
        # --- Arestas ---
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='#888'),
            opacity=0.3,
            mode='lines',
            hoverinfo='none'
        )

        # --- Nós ---
        node_x = []
        node_y = []
        node_z = []
        node_color = []
        node_text = []
        
        domain_colors = {
            'Physics': '#e74c3c',    # Red
            'Biology': '#2ecc71',    # Green
            'AI/ML': '#3498db',      # Blue
            'Math': '#9b59b6',       # Purple
            'InfoTheory': '#f1c40f', # Yellow
            'Other': '#95a5a6'       # Gray
        }
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            d = G.nodes[node]['domain']
            node_color.append(domain_colors.get(d, '#95a5a6'))
            
            # Hover text
            src = G.nodes[node]['source']
            txt = G.nodes[node]['content']
            node_text.append(f"<b>{d}</b><br>{src}<br><i>{txt}...</i>")

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=5,
                color=node_color,
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text'
        )

        # 5. Renderizar
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Rede Semântica de Conhecimento (Threshold > 0.7)',
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
                bgcolor='#0e1117'
            ),
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        output_file = "network_viz_3d.html"
        fig.write_html(output_file)
        logger.info(f"✅ Grafo interativo salvo em: {output_file}")

    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_network_graph()
