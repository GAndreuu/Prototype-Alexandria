#!/usr/bin/env python3
"""
Script de Calibração: VQVAEManifoldBridge
=========================================

Realiza Grid Search para encontrar os melhores parâmetros de deformação
do manifold, maximizando a correlação entre similaridade de cosseno (embedding)
e similaridade métrica (manifold).

Uso:
    python3 scripts/calibration/calibrate_bridge.py

Saída:
    config/bridge_calibration.json
"""

import os
import sys
import json
import numpy as np
from itertools import product
from datetime import datetime
from typing import List, Tuple, Dict
import logging

# Configurar path para importar módulos core
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Calibration")

try:
    from scipy.stats import pearsonr
except ImportError:
    logger.error("Scipy not found. Please install: pip install scipy")
    sys.exit(1)

try:
    from core.field.vqvae_manifold_bridge import VQVAEManifoldBridge, BridgeConfig, ProjectionMode
except ImportError:
    logger.error("Core modules not found. Run from project root.")
    sys.exit(1)

# =============================================================================
# MOCKS & DATA GENERATION
# =============================================================================

class MockVQVAE:
    """Mock simples do VQ-VAE para fornecer codebook."""
    def __init__(self, num_heads=4, codes_per_head=256, head_dim=128):
        self.codebook = np.random.randn(num_heads, codes_per_head, head_dim).astype(np.float32)
        # Normalizar para esfera unitária como embeddings reais
        norms = np.linalg.norm(self.codebook, axis=2, keepdims=True)
        self.codebook = self.codebook / (norms + 1e-8)
        
    def get_codebook(self):
        return self.codebook

def generate_synthetic_pairs(n_pairs=100, dim=384) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Gera pares de vetores com correlação controlada.
    
    Retorna:
        List[(vec_a, vec_b, target_similarity)]
    """
    pairs = []
    logger.info(f"Gerando {n_pairs} pares sintéticos...")
    
    for _ in range(n_pairs):
        # Bases aleatórias
        u = np.random.randn(dim)
        u /= np.linalg.norm(u)
        
        # Alvo de similaridade aleatório [-0.1, 0.9]
        target_sim = np.random.uniform(-0.1, 0.9)
        
        # Gerar v correlacionado com u
        # v = alpha * u + beta * orth_vec
        # sim = u . v = alpha
        alpha = target_sim
        
        # Vetor ortogonal
        orth = np.random.randn(dim)
        orth -= np.dot(orth, u) * u
        orth /= np.linalg.norm(orth)
        
        beta = np.sqrt(1 - alpha**2)
        v = alpha * u + beta * orth
        
        # Calcular sim real (check)
        real_sim = np.dot(u, v)
        pairs.append((u, v, real_sim))
        
    return pairs

def load_real_pairs(n_pairs=100) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Tenta carregar pares reais do LanceDB.
    Se falhar ou não tiver dados suficientes, retorna None.
    """
    try:
        from core.memory.storage import LanceDBStorage
        storage = LanceDBStorage()
        table = storage.get_table("alexandria_memory") # Nome hipotético
        
        if not table:
            return None
            
        data = table.search().limit(n_pairs * 2).to_list()
        
        if len(data) < 2:
            return None
            
        pairs = []
        vectors = [np.array(d['vector']) for d in data]
        
        # Gerar pares aleatórios dos dados reais
        import random
        for _ in range(n_pairs):
            v1 = random.choice(vectors)
            v2 = random.choice(vectors)
            
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            pairs.append((v1, v2, sim))
            
        logger.info(f"Carregados {len(pairs)} pares reais do LanceDB")
        return pairs
        
    except Exception as e:
        logger.debug(f"Não foi possível carregar dados reais: {e}")
        return None

# =============================================================================
# CALIBRATION LOGIC
# =============================================================================

def calibration_score(bridge, test_pairs):
    """
    Calcula correlação entre similaridade de cosseno e similaridade no manifold.
    """
    manifold_sims = []
    gt_sims = []
    
    for emb_a, emb_b, gt_sim in test_pairs:
        # Projetar no manifold
        point_a = bridge.embed(emb_a)
        point_b = bridge.embed(emb_b)
        
        # Distância no manifold deformado
        # Idealmente usaríamos geodesic distance, mas é lento.
        # Usamos Euclidean distance nas coordenadas projetadas como proxy rápido.
        dist = np.linalg.norm(point_a.coordinates - point_b.coordinates)
        
        # Converter distância -> similaridade (kernel RBF ou inverso simples)
        # 1 / (1 + d) mapeia [0, inf] -> [1, 0]
        sim_manifold = 1 / (1 + dist)
        
        manifold_sims.append(sim_manifold)
        gt_sims.append(gt_sim)
    
    # Pearson Check
    correlation, _ = pearsonr(manifold_sims, gt_sims)
    return correlation

def run_calibration():
    # 1. Obter Dados
    test_pairs = load_real_pairs()
    if not test_pairs:
        logger.warning("Usando dados SINTÉTICOS para calibração")
        test_pairs = generate_synthetic_pairs(n_pairs=200)
    
    # 2. Setup VQ-VAE
    # Tenta usar o real, fallback para Mock
    vqvae = MockVQVAE() 
    
    # 3. Grid Search
    param_grid = {
        'pull_strength': [0.1, 0.3, 0.5, 0.7],
        'pull_radius': [0.1, 0.3, 0.5],
        'deformation_strength': [0.1, 0.3, 0.5],
        'deformation_radius': [0.2, 0.3], # Mantenha pequeno para localidade
        'num_nearest_anchors': [4] # Fixado por heurística anterior
    }
    
    results = []
    total_combs = len(list(product(*param_grid.values())))
    logger.info(f"Iniciando Grid Search com {total_combs} combinações...")
    
    best_score = -1.0
    best_config = None
    
    for ps, pr, ds, dr, k in product(
        param_grid['pull_strength'],
        param_grid['pull_radius'],
        param_grid['deformation_strength'],
        param_grid['deformation_radius'],
        param_grid['num_nearest_anchors']
    ):
        config = BridgeConfig(
            pull_strength=ps,
            pull_radius=pr,
            deformation_strength=ds,
            deformation_radius=dr,
            num_nearest_anchors=k,
            projection_mode=ProjectionMode.WEIGHTED_ANCHORS
        )
        
        bridge = VQVAEManifoldBridge(config)
        bridge.connect_vqvae(vqvae)
        
        score = calibration_score(bridge, test_pairs)
        
        # Penalizar scores NaN (pode ocorrer se variância for 0)
        if np.isnan(score):
            score = -1.0
            
        logger.info(f"ps={ps} pr={pr} ds={ds} dr={dr} -> Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = {
                'pull_strength': ps,
                'pull_radius': pr,
                'deformation_strength': ds,
                'deformation_radius': dr,
                'num_nearest_anchors': k,
                'projection_mode': 'weighted',
                'calibration_score': score,
                'calibration_date': datetime.now().isoformat()
            }
    
    # 4. Salvar Resultados
    print("\n" + "="*50)
    print(f"MELHOR SCORE: {best_score:.4f}")
    print(f"CONFIG: {json.dumps(best_config, indent=2)}")
    print("="*50)
    
    os.makedirs('config', exist_ok=True)
    with open('config/bridge_calibration.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    logger.info("Configuração salva em config/bridge_calibration.json")

if __name__ == "__main__":
    run_calibration()
