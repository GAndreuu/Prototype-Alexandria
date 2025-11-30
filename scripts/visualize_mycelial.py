#!/usr/bin/env python3
"""
Alexandria :: Visualização do MycelialReasoning

Gera visualizações interativas da rede micelial.

Uso:
    python visualize_mycelial.py                    # Dashboard completo
    python visualize_mycelial.py --live             # Visualização em tempo real durante treino
    python visualize_mycelial.py --graph            # Só o grafo 3D
    python visualize_mycelial.py --heatmap          # Só os heatmaps
    python visualize_mycelial.py --export imgs/     # Exportar imagens
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import time

# Path do projeto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports de visualização
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly não disponível. Instale com: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Import do projeto
try:
    from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False
    print("⚠️  MycelialReasoning não disponível")


# =============================================================================
# VISUALIZADOR
# =============================================================================

class MycelialVisualizer:
    """Gera visualizações da rede micelial."""
    
    def __init__(self, mycelial: MycelialReasoning = None):
        if mycelial is None:
            # Carregar estado salvo
            config = MycelialConfig(save_path=str(PROJECT_ROOT / "data" / "mycelial_state.npz"))
            self.mycelial = MycelialReasoning(config)
        else:
            self.mycelial = mycelial
        
        # Cores por head
        self.head_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        self.head_names = ['Head 0', 'Head 1', 'Head 2', 'Head 3']
    
    # =========================================================================
    # HEATMAPS
    # =========================================================================
    
    def plot_connection_heatmaps(self, show: bool = True) -> go.Figure:
        """
        Heatmap das matrizes de conexão por head.
        
        Mostra padrões de conectividade.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Head {i} - Conexões' for i in range(4)],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )
        
        for h in range(4):
            row = h // 2 + 1
            col = h % 2 + 1
            
            # Matriz de conexões
            matrix = self.mycelial.connections[h]
            
            # Criar heatmap
            heatmap = go.Heatmap(
                z=matrix,
                colorscale='Viridis',
                showscale=(h == 3),  # Só mostrar escala no último
                name=f'Head {h}'
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title='Matrizes de Conexão por Head',
            height=800,
            width=1000,
            template='plotly_dark'
        )
        
        if show:
            fig.show()
        
        return fig
    
    def plot_activation_heatmap(self, show: bool = True) -> go.Figure:
        """Heatmap de contagem de ativações."""
        fig = go.Figure()
        
        # Stack activation counts
        activations = self.mycelial.activation_counts
        
        fig.add_trace(go.Heatmap(
            z=activations,
            x=list(range(256)),
            y=['Head 0', 'Head 1', 'Head 2', 'Head 3'],
            colorscale='Hot',
            colorbar=dict(title='Ativações')
        ))
        
        fig.update_layout(
            title='Mapa de Ativação por Código',
            xaxis_title='Código (0-255)',
            yaxis_title='Head',
            height=400,
            width=1200,
            template='plotly_dark'
        )
        
        if show:
            fig.show()
        
        return fig
    
    # =========================================================================
    # GRAFO 3D
    # =========================================================================
    
    def plot_network_graph_3d(
        self, 
        head: int = 0,
        threshold: float = 0.1,
        max_edges: int = 500,
        show: bool = True
    ) -> go.Figure:
        """
        Grafo 3D das conexões entre códigos.
        
        Args:
            head: Qual head visualizar
            threshold: Força mínima para mostrar conexão
            max_edges: Limite de arestas para performance
        """
        connections = self.mycelial.connections[head]
        activations = self.mycelial.activation_counts[head]
        
        # Encontrar conexões acima do threshold
        edges = []
        strengths = []
        
        for i in range(256):
            for j in range(256):
                if connections[i, j] > threshold:
                    edges.append((i, j))
                    strengths.append(connections[i, j])
        
        # Limitar para performance
        if len(edges) > max_edges:
            # Pegar as mais fortes
            indices = np.argsort(strengths)[-max_edges:]
            edges = [edges[i] for i in indices]
            strengths = [strengths[i] for i in indices]
        
        if len(edges) == 0:
            print(f"⚠️  Nenhuma conexão acima do threshold {threshold}")
            return None
        
        # Layout: posicionar códigos em círculo no plano XY, Z baseado em ativação
        angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        radius = 10
        
        x_pos = radius * np.cos(angles)
        y_pos = radius * np.sin(angles)
        z_pos = np.log1p(activations)  # Log para suavizar
        z_pos = z_pos / (z_pos.max() + 1e-8) * 5  # Normalizar
        
        # Criar traces para arestas
        edge_x, edge_y, edge_z = [], [], []
        edge_colors = []
        
        for (i, j), strength in zip(edges, strengths):
            edge_x.extend([x_pos[i], x_pos[j], None])
            edge_y.extend([y_pos[i], y_pos[j], None])
            edge_z.extend([z_pos[i], z_pos[j], None])
            edge_colors.append(strength)
        
        # Trace de arestas
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(150, 150, 150, 0.3)',
                width=1
            ),
            hoverinfo='none',
            name='Conexões'
        )
        
        # Trace de nós
        node_sizes = np.log1p(activations) * 3 + 3
        node_colors = activations
        
        node_trace = go.Scatter3d(
            x=x_pos, y=y_pos, z=z_pos,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Plasma',
                colorbar=dict(title='Ativações'),
                line=dict(width=0.5, color='white')
            ),
            text=[f'Código {i}<br>Ativações: {activations[i]}' for i in range(256)],
            hoverinfo='text',
            name='Códigos'
        )
        
        # Destacar hubs
        hubs = self.mycelial.get_hub_codes(10)
        hub_codes = [h['code'] for h in hubs if h['head'] == head]
        
        if hub_codes:
            hub_trace = go.Scatter3d(
                x=x_pos[hub_codes],
                y=y_pos[hub_codes],
                z=z_pos[hub_codes],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='gold',
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=[str(c) for c in hub_codes],
                textposition='top center',
                name='Hubs'
            )
        else:
            hub_trace = None
        
        # Criar figura
        fig = go.Figure(data=[edge_trace, node_trace])
        if hub_trace:
            fig.add_trace(hub_trace)
        
        fig.update_layout(
            title=f'Rede Micelial - Head {head} ({len(edges)} conexões)',
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(title='Ativação (log)', showgrid=True),
                bgcolor='rgb(20, 20, 30)'
            ),
            height=800,
            width=1000,
            template='plotly_dark',
            showlegend=True
        )
        
        if show:
            fig.show()
        
        return fig
    
    def plot_all_heads_graph(
        self,
        threshold: float = 0.2,
        max_edges_per_head: int = 100,
        show: bool = True
    ) -> go.Figure:
        """Grafo 3D com todos os 4 heads separados."""
        
        fig = go.Figure()
        
        offsets = [(0, 0), (25, 0), (0, 25), (25, 25)]
        
        for h in range(4):
            connections = self.mycelial.connections[h]
            activations = self.mycelial.activation_counts[h]
            
            ox, oy = offsets[h]
            
            # Posições
            angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
            radius = 8
            
            x_pos = ox + radius * np.cos(angles)
            y_pos = oy + radius * np.sin(angles)
            z_pos = np.log1p(activations)
            z_pos = z_pos / (z_pos.max() + 1e-8) * 5
            
            # Arestas
            edges = []
            strengths = []
            
            for i in range(256):
                for j in range(256):
                    if connections[i, j] > threshold:
                        edges.append((i, j))
                        strengths.append(connections[i, j])
            
            # Limitar
            if len(edges) > max_edges_per_head:
                indices = np.argsort(strengths)[-max_edges_per_head:]
                edges = [edges[i] for i in indices]
            
            # Trace arestas
            edge_x, edge_y, edge_z = [], [], []
            for (i, j) in edges:
                edge_x.extend([x_pos[i], x_pos[j], None])
                edge_y.extend([y_pos[i], y_pos[j], None])
                edge_z.extend([z_pos[i], z_pos[j], None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=self.head_colors[h], width=1),
                opacity=0.3,
                name=f'Head {h} edges',
                showlegend=False
            ))
            
            # Trace nós
            fig.add_trace(go.Scatter3d(
                x=x_pos, y=y_pos, z=z_pos,
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.head_colors[h],
                ),
                name=f'Head {h}',
                text=[f'H{h} C{i}: {activations[i]}' for i in range(256)],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title='Rede Micelial - Todos os Heads',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(title='Ativação'),
                bgcolor='rgb(20, 20, 30)'
            ),
            height=800,
            width=1000,
            template='plotly_dark'
        )
        
        if show:
            fig.show()
        
        return fig
    
    # =========================================================================
    # ESTATÍSTICAS
    # =========================================================================
    
    def plot_statistics(self, show: bool = True) -> go.Figure:
        """Dashboard de estatísticas."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribuição de Ativações',
                'Distribuição de Força de Conexão',
                'Grau de Conectividade por Head',
                'Top 20 Códigos Mais Ativos'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Distribuição de ativações
        all_activations = self.mycelial.activation_counts.flatten()
        all_activations = all_activations[all_activations > 0]
        
        fig.add_trace(
            go.Histogram(x=all_activations, nbinsx=50, name='Ativações'),
            row=1, col=1
        )
        
        # 2. Distribuição de força de conexão
        all_connections = self.mycelial.connections.flatten()
        all_connections = all_connections[all_connections > 0]
        
        fig.add_trace(
            go.Histogram(x=all_connections, nbinsx=50, name='Força'),
            row=1, col=2
        )
        
        # 3. Grau de conectividade por head
        degrees = []
        heads = []
        for h in range(4):
            degree = np.sum(self.mycelial.connections[h] > 0.05, axis=1)
            degrees.extend(degree[degree > 0].tolist())
            heads.extend([f'Head {h}'] * len(degree[degree > 0]))
        
        fig.add_trace(
            go.Box(x=heads, y=degrees, name='Grau'),
            row=2, col=1
        )
        
        # 4. Top 20 códigos mais ativos
        hubs = self.mycelial.get_hub_codes(20)
        hub_names = [f"H{h['head']}C{h['code']}" for h in hubs]
        hub_counts = [h['activation_count'] for h in hubs]
        
        fig.add_trace(
            go.Bar(x=hub_names, y=hub_counts, name='Ativações',
                   marker_color=[self.head_colors[h['head']] for h in hubs]),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            width=1200,
            template='plotly_dark',
            title='Estatísticas da Rede Micelial',
            showlegend=False
        )
        
        if show:
            fig.show()
        
        return fig
    
    # =========================================================================
    # DASHBOARD COMPLETO
    # =========================================================================
    
    def create_dashboard(self, export_path: Optional[str] = None):
        """Cria dashboard completo com todas as visualizações."""
        
        print("=" * 60)
        print("ALEXANDRIA :: VISUALIZAÇÃO MICELIAL")
        print("=" * 60)
        
        stats = self.mycelial.get_network_stats()
        print(f"\n📊 Observações: {stats['total_observations']:,}")
        print(f"📊 Conexões ativas: {stats['active_connections']:,}")
        print(f"📊 Densidade: {stats['density']:.4%}")
        
        print("\n⏳ Gerando visualizações...")
        
        # Estatísticas
        print("   [1/4] Estatísticas...")
        fig_stats = self.plot_statistics(show=False)
        
        # Heatmap de ativações
        print("   [2/4] Mapa de ativações...")
        fig_activation = self.plot_activation_heatmap(show=False)
        
        # Grafo 3D (head 0)
        print("   [3/4] Grafo 3D...")
        fig_graph = self.plot_network_graph_3d(head=0, show=False)
        
        # Heatmaps de conexão
        print("   [4/4] Heatmaps de conexão...")
        fig_heatmaps = self.plot_connection_heatmaps(show=False)
        
        # Exportar se pedido
        if export_path:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            fig_stats.write_html(str(export_dir / "stats.html"))
            fig_activation.write_html(str(export_dir / "activation.html"))
            if fig_graph:
                fig_graph.write_html(str(export_dir / "graph_3d.html"))
            fig_heatmaps.write_html(str(export_dir / "heatmaps.html"))
            
            print(f"\n✅ Exportado para {export_path}/")
        
        # Mostrar
        print("\n🎨 Abrindo visualizações no navegador...")
        fig_stats.show()
        fig_activation.show()
        if fig_graph:
            fig_graph.show()
        
        print("\n✅ Concluído!")


# =============================================================================
# VISUALIZAÇÃO AO VIVO
# =============================================================================

class LiveVisualizer:
    """Visualização em tempo real durante treinamento."""
    
    def __init__(self, mycelial: MycelialReasoning):
        self.mycelial = mycelial
        self.history = {
            'step': [],
            'connections': [],
            'density': [],
            'max_strength': [],
        }
        self.running = False
    
    def update(self):
        """Atualiza histórico."""
        stats = self.mycelial.get_network_stats()
        self.history['step'].append(stats['step'])
        self.history['connections'].append(stats['active_connections'])
        self.history['density'].append(stats['density'])
        self.history['max_strength'].append(stats['max_connection_strength'])
    
    def plot_live(self, interval_ms: int = 1000):
        """
        Plota métricas em tempo real.
        
        Chame isso em um thread separado ou use com notebook.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib necessário para visualização ao vivo")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('MycelialReasoning - Treino ao Vivo')
        
        plt.ion()
        
        self.running = True
        
        while self.running:
            self.update()
            
            # Limpar
            for ax in axes.flat:
                ax.clear()
            
            steps = self.history['step']
            
            # Conexões
            axes[0, 0].plot(steps, self.history['connections'], 'b-')
            axes[0, 0].set_title('Conexões Ativas')
            axes[0, 0].set_xlabel('Step')
            
            # Densidade
            axes[0, 1].plot(steps, self.history['density'], 'g-')
            axes[0, 1].set_title('Densidade')
            axes[0, 1].set_xlabel('Step')
            
            # Força máxima
            axes[1, 0].plot(steps, self.history['max_strength'], 'r-')
            axes[1, 0].set_title('Força Máxima')
            axes[1, 0].set_xlabel('Step')
            
            # Heatmap atual (head 0)
            im = axes[1, 1].imshow(
                self.mycelial.connections[0],
                cmap='viridis',
                aspect='auto'
            )
            axes[1, 1].set_title('Conexões Head 0')
            
            plt.tight_layout()
            plt.pause(interval_ms / 1000)
        
        plt.ioff()
        plt.show()
    
    def stop(self):
        """Para visualização."""
        self.running = False


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualiza MycelialReasoning")
    parser.add_argument("--graph", action="store_true", help="Só grafo 3D")
    parser.add_argument("--heatmap", action="store_true", help="Só heatmaps")
    parser.add_argument("--stats", action="store_true", help="Só estatísticas")
    parser.add_argument("--all-heads", action="store_true", help="Grafo com todos os heads")
    parser.add_argument("--head", type=int, default=0, help="Qual head visualizar")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold de conexão")
    parser.add_argument("--export", type=str, help="Exportar para diretório")
    parser.add_argument("--live", action="store_true", help="Visualização ao vivo")
    args = parser.parse_args()
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly necessário. Instale com: pip install plotly")
        return 1
    
    if not MYCELIAL_AVAILABLE:
        print("❌ MycelialReasoning não encontrado")
        return 1
    
    viz = MycelialVisualizer()
    
    if args.live:
        print("⚠️  Modo ao vivo requer treino rodando em paralelo")
        print("    Use em notebook ou integre com train_mycelial.py")
        return 0
    
    if args.graph:
        viz.plot_network_graph_3d(head=args.head, threshold=args.threshold)
    elif args.all_heads:
        viz.plot_all_heads_graph(threshold=args.threshold)
    elif args.heatmap:
        viz.plot_connection_heatmaps()
        viz.plot_activation_heatmap()
    elif args.stats:
        viz.plot_statistics()
    else:
        viz.create_dashboard(export_path=args.export)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
