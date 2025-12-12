#!/usr/bin/env python3
"""
Alexandria :: Visualização da Estrutura do Projeto
Gera visualizações 2D, 3D e 4D (animadas) da estrutura de arquivos.
Otimizado para grandes diretórios.
"""

import os
import sys
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple

# Path do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "structure_viz"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Diretórios a ignorar
EXCLUDE_DIRS = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', '.benchmarks'}
# Diretórios grandes para não detalhar arquivos no 3D
LARGE_DIRS = {'data', 'archive'}

def build_graph(root_dir: Path) -> nx.DiGraph:
    """Constrói um grafo direcionado da estrutura de arquivos."""
    G = nx.DiGraph()
    
    # Adicionar nó raiz
    root_name = root_dir.name
    G.add_node(root_name, type='root', size=10, path=str(root_dir))
    
    print(f"📂 Escaneando {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        # Filtrar diretórios
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        current_path = Path(root)
        rel_current = current_path.relative_to(root_dir)
        
        # Se estiver dentro de um diretório grande, pular arquivos
        if any(part in LARGE_DIRS for part in rel_current.parts):
            files = [] # Ignorar arquivos em pastas de dados
        
        current_id = str(rel_current) if current_path != root_dir else root_name
        
        # Adicionar diretórios
        for d in dirs:
            dir_path = current_path / d
            rel_path = dir_path.relative_to(root_dir)
            unique_id = str(rel_path)
            
            G.add_node(unique_id, type='directory', size=5, label=d)
            G.add_edge(current_id, unique_id)
            
        # Adicionar arquivos
        for f in files:
            if f.startswith('.'): continue
                
            file_path = current_path / f
            rel_path = file_path.relative_to(root_dir)
            unique_id = str(rel_path)
            
            # Tamanho
            try:
                size = file_path.stat().st_size
                size_score = max(2, min(10, len(str(size))))
            except:
                size_score = 2
                
            G.add_node(unique_id, type='file', size=size_score, label=f)
            G.add_edge(current_id, unique_id)
            
    return G

def plot_2d_treemap(root_dir: Path):
    """Gera um Treemap 2D da estrutura."""
    print("   [2D] Gerando Treemap...")
    data = []
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        for f in files:
            if f.startswith('.'): continue
            
            file_path = Path(root) / f
            try:
                size = file_path.stat().st_size
            except:
                size = 0
                
            rel_path = file_path.relative_to(root_dir)
            
            # Adicionar hierarquia
            data.append({
                'path': str(rel_path),
                'file': f,
                'parent': str(Path(root).relative_to(root_dir)) if root != str(root_dir) else root_dir.name,
                'size': size,
                'type': f.split('.')[-1] if '.' in f else 'no_ext'
            })
            
    if not data:
        print("⚠️  Nenhum dado encontrado para treemap.")
        return

    fig = px.treemap(
        data,
        path=[px.Constant(root_dir.name), 'type', 'file'],
        values='size',
        color='type',
        title='Mapa de Arquivos 2D (Tamanho por Tipo)',
        template='plotly_dark'
    )
    
    output_path = REPORTS_DIR / "structure_2d_treemap.html"
    fig.write_html(str(output_path))
    print(f"   ✅ Salvo em: {output_path}")

def plot_3d_network(G: nx.DiGraph):
    """Gera um grafo 3D interativo."""
    print(f"   [3D] Gerando Grafo ({len(G.nodes)} nós)...")
    
    # Limitar nós se for muito grande
    MAX_NODES = 2000
    if len(G.nodes) > MAX_NODES:
        print(f"   ⚠️  Muitos nós ({len(G.nodes)}). Filtrando para visualização 3D...")
        # Manter diretórios e arquivos próximos da raiz
        nodes_to_keep = {n for n, d in G.nodes(data=True) if d.get('type') in ('root', 'directory')}
        # Adicionar alguns arquivos
        files = [n for n, d in G.nodes(data=True) if d.get('type') == 'file']
        nodes_to_keep.update(files[:MAX_NODES - len(nodes_to_keep)])
        
        H = G.subgraph(nodes_to_keep)
    else:
        H = G

    # Layout 3D spring
    print("   [3D] Calculando layout...")
    pos = nx.spring_layout(H, dim=3, seed=42, iterations=50)
    
    # Extrair posições
    x_nodes = [pos[k][0] for k in H.nodes]
    y_nodes = [pos[k][1] for k in H.nodes]
    z_nodes = [pos[k][2] for k in H.nodes]
    
    # Cores e tamanhos
    node_colors = []
    node_sizes = []
    node_texts = []
    
    for node in H.nodes:
        node_type = H.nodes[node].get('type', 'file')
        label = H.nodes[node].get('label', node)
        
        if node_type == 'root':
            node_colors.append('#FF6B6B') # Red
            node_sizes.append(15)
        elif node_type == 'directory':
            node_colors.append('#4ECDC4') # Teal
            node_sizes.append(8)
        else:
            node_colors.append('#45B7D1') # Blue
            node_sizes.append(4)
            
        node_texts.append(f"{label} ({node_type})")
        
    # Arestas
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in H.edges:
        if edge[0] in pos and edge[1] in pos:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
    # Criar traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#888', width=1),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=0)
        ),
        text=node_texts,
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f'Estrutura de Arquivos 3D ({len(H.nodes)} nós)',
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgb(20, 20, 30)'
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_path = REPORTS_DIR / "structure_3d_network.html"
    fig.write_html(str(output_path))
    print(f"   ✅ Salvo em: {output_path}")
    
    return H, pos

def plot_4d_animation(H: nx.DiGraph, pos: Dict):
    """Gera uma animação '4D' (rotação temporal) do grafo."""
    print("   [4D] Gerando Animação...")
    
    x_nodes = [pos[k][0] for k in H.nodes]
    y_nodes = [pos[k][1] for k in H.nodes]
    z_nodes = [pos[k][2] for k in H.nodes]
    
    # Cores
    node_colors = []
    for node in H.nodes:
        t = H.nodes[node].get('type', 'file')
        if t == 'root': node_colors.append(1)
        elif t == 'directory': node_colors.append(0.5)
        else: node_colors.append(0)

    # Criar figura base
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(
                size=5,
                color=node_colors,
                colorscale='Viridis'
            ),
            text=[H.nodes[n].get('label', n) for n in H.nodes]
        )]
    )
    
    # Adicionar arestas
    edge_x, edge_y, edge_z = [], [], []
    for edge in H.edges:
        if edge[0] in pos and edge[1] in pos:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(200,200,200,0.2)', width=1),
        hoverinfo='none'
    ))

    # Configurar animação
    fig.update_layout(
        title='Estrutura 4D (Animação)',
        scene_camera_eye=dict(x=-1.25, y=-1.25, z=0.125),
        template='plotly_dark',
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1,
            x=0.8,
            xanchor='left',
            yanchor='bottom',
            pad=dict(t=45, r=10),
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=True), 
                                fromcurrent=True, 
                                mode='immediate')]
            )]
        )]
    )
    
    # Gerar frames
    frames = []
    import numpy as np
    
    for t in np.linspace(0, 6.28, 60):
        xe = 1.5 * np.cos(t)
        ye = 1.5 * np.sin(t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=0.5))))
        
    fig.frames = frames
    
    output_path = REPORTS_DIR / "structure_4d_anim.html"
    fig.write_html(str(output_path))
    print(f"   ✅ Salvo em: {output_path}")

def main():
    print("🔍 Analisando estrutura de arquivos...")
    G = build_graph(PROJECT_ROOT)
    print(f"📊 Encontrados {len(G.nodes)} nós e {len(G.edges)} conexões.")
    
    print("\n🎨 Gerando visualizações...")
    plot_2d_treemap(PROJECT_ROOT)
    H, pos = plot_3d_network(G)
    plot_4d_animation(H, pos)
    
    print("\n✨ Processo concluído! Verifique a pasta 'reports/structure_viz/'.")

if __name__ == "__main__":
    main()
