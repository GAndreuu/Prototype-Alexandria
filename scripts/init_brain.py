import sys
import os
import numpy as np

# Adicionar diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.topology_engine import TopologyEngine
from config import settings

def init_brain():
    print("🧠 Inicializando Cérebro (Topology Engine)...")
    
    # 1. Inicializar Engine
    engine = TopologyEngine()
    
    # 2. Gerar dados iniciais (se não houver dados reais ainda)
    # Simulando alguns vetores para criar o manifold inicial
    print("⚡ Gerando vetores iniciais...")
    # Criar 100 vetores aleatórios de 384 dimensões para garantir que o K-Means funcione
    initial_vectors = np.random.randn(100, 384)
    
    # 3. Treinar Manifold
    print("🏋️ Treinando Manifold...")
    # Usar n_clusters menor para inicialização se tiver poucos dados
    n_clusters = min(32, settings.N_CLUSTERS)
    engine.train_manifold(initial_vectors, n_clusters=n_clusters)
    
    # 4. Salvar
    print(f"💾 Salvando em {settings.TOPOLOGY_FILE}...")
    engine.save_topology(settings.TOPOLOGY_FILE)
    
    print("✅ Cérebro Inicializado com Sucesso!")
    print("Agora o sistema pode iniciar sem erros de 'Cérebro não inicializado'.")

if __name__ == "__main__":
    init_brain()
