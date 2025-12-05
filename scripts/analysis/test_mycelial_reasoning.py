#!/usr/bin/env python3
"""
Teste do Mycelial Reasoning após Bootstrap
==========================================

Verifica se a rede aprendeu conexões úteis e se o reasoning melhorou.

Uso:
    python test_mycelial_reasoning.py --state-path ./data/mycelial_state.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

def load_mycelial_state(state_path: str) -> dict:
    """Carrega estado da rede mycelial."""
    print(f"Carregando estado de: {state_path}")
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    
    # Adapter for Sparse Graph -> Dense Connections
    if 'graph' in state and 'connections' not in state:
        print("Detectado formato Sparse Graph. Convertendo para Dense Matrix para análise...")
        graph = state['graph']
        num_heads = 4 # Default
        if 'config' in state:
            num_heads = state['config'].get('num_heads', 4)
        
        codebook_size = 256
        if 'config' in state:
            codebook_size = state['config'].get('codebook_size', 256)
            
        connections = [np.zeros((codebook_size, codebook_size)) for _ in range(num_heads)]
        
        for node_id, edges in graph.items():
            # Handle tuple keys (head, code)
            if isinstance(node_id, tuple):
                head_idx, code_idx = node_id
            elif isinstance(node_id, str):
                try:
                    h_str, c_str = node_id.split('_')
                    head_idx = int(h_str[1:])
                    code_idx = int(c_str)
                except ValueError:
                    continue
            else:
                continue
                
            for target_id, edge_data in edges.items():
                # Handle target tuple
                if isinstance(target_id, tuple):
                    target_head, target_code = target_id
                elif isinstance(target_id, str):
                    try:
                        t_h_str, t_c_str = target_id.split('_')
                        target_code = int(t_c_str)
                    except ValueError:
                        continue
                else:
                    continue
                
                # Debug edge data
                if not isinstance(edge_data, dict):
                    # Maybe it's just the weight directly?
                    weight = edge_data
                else:
                    weight = edge_data.get('weight', 0.0)
                
                if weight > connections[head_idx][code_idx, target_code]:
                    connections[head_idx][code_idx, target_code] = weight
                
        state['connections'] = connections

    return state


def analyze_network_health(state: dict) -> dict:
    """Analisa saúde geral da rede."""
    
    connections = state.get('connections', [])
    if isinstance(connections[0], list):
        connections = [np.array(c) for c in connections]
    
    stats = {
        'num_heads': len(connections),
        'codebook_size': connections[0].shape[0] if len(connections) > 0 else 0,
        'total_observations': state.get('total_observations', 0),
        'per_head': []
    }
    
    total_active = 0
    total_possible = 0
    
    for h, conn in enumerate(connections):
        active = np.sum(conn > 0.01)
        possible = conn.shape[0] * conn.shape[1]
        density = active / possible if possible > 0 else 0
        
        # Estatísticas de peso
        nonzero = conn[conn > 0.01]
        
        head_stats = {
            'head': h,
            'active_connections': int(active),
            'density': density,
            'mean_weight': float(np.mean(nonzero)) if len(nonzero) > 0 else 0,
            'max_weight': float(np.max(conn)),
            'std_weight': float(np.std(nonzero)) if len(nonzero) > 0 else 0,
        }
        stats['per_head'].append(head_stats)
        
        total_active += active
        total_possible += possible
    
    stats['total_active_connections'] = total_active
    stats['global_density'] = total_active / total_possible if total_possible > 0 else 0
    
    return stats


def find_strongest_connections(state: dict, top_k: int = 20) -> List[Dict]:
    """Encontra as conexões mais fortes na rede."""
    
    # If we have the raw graph, use it for better accuracy
    if 'graph' in state:
        graph = state['graph']
        all_edges = []
        for node_id, edges in graph.items():
            for target_id, data in edges.items():
                weight = data['weight'] if isinstance(data, dict) else data
                all_edges.append({
                    'from': node_id,
                    'to': target_id,
                    'weight': weight
                })
        
        all_edges.sort(key=lambda x: x['weight'], reverse=True)
        
        # Format for display
        formatted = []
        for edge in all_edges[:top_k]:
            # Parse IDs
            try:
                if isinstance(edge['from'], tuple):
                    h_from, c_from = edge['from']
                else:
                    h_from_str, c_from_str = edge['from'].split('_')
                    h_from = int(h_from_str[1:])
                    c_from = int(c_from_str)
                
                if isinstance(edge['to'], tuple):
                    h_to, c_to = edge['to']
                else:
                    h_to_str, c_to_str = edge['to'].split('_')
                    h_to = int(h_to_str[1:])
                    c_to = int(c_to_str)
                    
                formatted.append({
                    'head': h_from,
                    'from_code': c_from,
                    'to_code': f"{h_to}_{c_to}", # Show full target ID
                    'weight': edge['weight'] if isinstance(edge, dict) else edge
                })
            except:
                continue
        return formatted

    connections = state.get('connections', [])
    if isinstance(connections[0], list):
        connections = [np.array(c) for c in connections]
    
    all_connections = []
    
    for h, conn in enumerate(connections):
        # Encontrar top conexões neste head
        flat_indices = np.argsort(conn.flatten())[-top_k*2:][::-1]
        
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, conn.shape)
            weight = conn[i, j]
            
            if weight > 0.01:
                all_connections.append({
                    'head': h,
                    'from_code': int(i),
                    'to_code': int(j),
                    'weight': float(weight)
                })
    
    # Ordenar por peso e pegar top_k
    all_connections.sort(key=lambda x: x['weight'], reverse=True)
    return all_connections[:top_k]


def find_hub_codes(state: dict, top_k: int = 5) -> Dict[int, List[Dict]]:
    """Identifica códigos hub (mais conectados) por head."""
    
    connections = state.get('connections', [])
    if isinstance(connections[0], list):
        connections = [np.array(c) for c in connections]
    
    hubs_per_head = {}
    
    for h, conn in enumerate(connections):
        # Grau de saída (soma das conexões de cada código)
        out_degree = np.sum(conn, axis=1)
        # Grau de entrada
        in_degree = np.sum(conn, axis=0)
        # Grau total
        total_degree = out_degree + in_degree
        
        top_indices = np.argsort(total_degree)[-top_k:][::-1]
        
        hubs_per_head[h] = [
            {
                'code': int(idx),
                'total_degree': float(total_degree[idx]),
                'out_degree': float(out_degree[idx]),
                'in_degree': float(in_degree[idx])
            }
            for idx in top_indices
        ]
    
    return hubs_per_head


def propagate_activation(
    state: dict,
    initial_codes: List[int],
    steps: int = 3,
    threshold: float = 0.1
) -> Dict[int, np.ndarray]:
    """
    Simula propagação de ativação na rede.
    """
    # Use sparse graph propagation if available
    if 'graph' in state:
        graph = state['graph']
        activations = defaultdict(float)
        
        # Set initial
        for h, code in enumerate(initial_codes):
            activations[(h, code)] = 1.0
            
        for step in range(steps):
            new_activations = defaultdict(float)
            
            # Decay old
            for node, val in activations.items():
                new_activations[node] += val * 0.5
            
            # Propagate
            for node, val in activations.items():
                if val < threshold: continue
                if node in graph:
                    for target, data in graph[node].items():
                        weight = data['weight'] if isinstance(data, dict) else data
                        new_activations[target] += val * weight
            
            # Normalize per head? Or global?
            # Simple max normalization
            max_val = max(new_activations.values()) if new_activations else 0
            if max_val > 1.0:
                for k in new_activations:
                    new_activations[k] /= max_val
            
            activations = new_activations
            
        # Convert to per-head arrays
        final_activations = {}
        for h in range(4):
            final_activations[h] = np.zeros(256)
            
        for node, val in activations.items():
            try:
                if isinstance(node, tuple):
                    h, c = node
                else:
                    h_str, c_str = node.split('_')
                    h = int(h_str[1:])
                    c = int(c_str)
                
                if val > threshold:
                    final_activations[h][c] = val
            except:
                pass
                
        return final_activations

    # Fallback to dense
    connections = state.get('connections', [])
    if isinstance(connections[0], list):
        connections = [np.array(c) for c in connections]
    
    num_heads = len(connections)
    codebook_size = connections[0].shape[0]
    
    # Inicializar ativações
    activations = {}
    for h in range(num_heads):
        activations[h] = np.zeros(codebook_size)
        if h < len(initial_codes):
            activations[h][initial_codes[h]] = 1.0
    
    # Propagar
    for step in range(steps):
        new_activations = {}
        
        for h in range(num_heads):
            # Propagação intra-head
            spread = activations[h] @ connections[h]
            
            # Normalizar
            if np.max(spread) > 0:
                spread = spread / np.max(spread)
            
            # Combinar com ativação anterior (decay)
            new_activations[h] = activations[h] * 0.5 + spread * 0.5
            
            # Aplicar threshold
            new_activations[h][new_activations[h] < threshold] = 0
        
        activations = new_activations
    
    return activations


def get_activated_codes(activations: Dict[int, np.ndarray], top_k: int = 5) -> Dict[int, List[Tuple[int, float]]]:
    """Extrai os códigos mais ativados após propagação."""
    
    result = {}
    for h, act in activations.items():
        top_indices = np.argsort(act)[-top_k:][::-1]
        result[h] = [
            (int(idx), float(act[idx]))
            for idx in top_indices
            if act[idx] > 0.01
        ]
    
    return result


def test_semantic_association(state: dict, test_cases: List[Dict]) -> List[Dict]:
    """
    Testa associação semântica: dado um padrão inicial,
    que outros padrões a rede sugere?
    """
    
    results = []
    
    for test in test_cases:
        name = test.get('name', 'unnamed')
        initial = test['initial_codes']
        
        print(f"\n--- Teste: {name} ---")
        print(f"Códigos iniciais: {initial}")
        
        # Propagar
        activations = propagate_activation(state, initial, steps=3)
        activated = get_activated_codes(activations, top_k=5)
        
        print("Códigos ativados após propagação:")
        for h, codes in activated.items():
            if codes:
                codes_str = ", ".join([f"{c}({w:.3f})" for c, w in codes])
                print(f"  Head {h}: {codes_str}")
        
        results.append({
            'name': name,
            'initial': initial,
            'activated': activated
        })
    
    return results


def compare_with_random(state: dict, num_samples: int = 100) -> dict:
    """
    Compara propagação de padrões reais vs. aleatórios.
    
    Se a rede aprendeu estrutura real, padrões reais devem
    ativar mais códigos relacionados que padrões aleatórios.
    """
    
    connections = state.get('connections', [])
    codebook_size = 256
    
    # Gerar padrões aleatórios
    random_patterns = np.random.randint(0, codebook_size, size=(num_samples, 4))
    
    random_activations = []
    for pattern in random_patterns:
        act = propagate_activation(state, pattern.tolist(), steps=3)
        total_activation = sum(np.sum(a) for a in act.values())
        random_activations.append(total_activation)
    
    return {
        'mean_random_activation': float(np.mean(random_activations)),
        'std_random_activation': float(np.std(random_activations)),
        'max_random_activation': float(np.max(random_activations)),
        'min_random_activation': float(np.min(random_activations)),
    }


def print_report(health: dict, strongest: List[Dict], hubs: Dict, random_baseline: dict):
    """Imprime relatório completo."""
    
    print("\n" + "="*70)
    print("RELATÓRIO DE SAÚDE DA REDE MYCELIAL")
    print("="*70)
    
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Observações treinadas: {health['total_observations']:,}")
    print(f"   Conexões ativas: {health['total_active_connections']:,}")
    print(f"   Densidade global: {health['global_density']*100:.2f}%")
    
    print(f"\n📈 POR HEAD:")
    for h_stats in health['per_head']:
        print(f"   Head {h_stats['head']}:")
        print(f"      Conexões ativas: {h_stats['active_connections']:,}")
        print(f"      Densidade: {h_stats['density']*100:.2f}%")
        print(f"      Peso médio: {h_stats['mean_weight']:.4f}")
        print(f"      Peso máximo: {h_stats['max_weight']:.4f}")
    
    print(f"\n🔗 TOP 10 CONEXÕES MAIS FORTES:")
    for i, conn in enumerate(strongest[:10]):
        print(f"   {i+1}. Head {conn['head']}: "
              f"Código {conn['from_code']} → {conn['to_code']} "
              f"(peso: {conn['weight']:.4f})")
    
    print(f"\n🎯 CÓDIGOS HUB (mais conectados):")
    for h, hub_list in hubs.items():
        codes = [f"{hub['code']}(grau:{hub['total_degree']:.2f})" for hub in hub_list[:3]]
        print(f"   Head {h}: {', '.join(codes)}")
    
    print(f"\n📉 BASELINE ALEATÓRIO:")
    print(f"   Ativação média (random): {random_baseline['mean_random_activation']:.2f}")
    print(f"   Desvio padrão: {random_baseline['std_random_activation']:.2f}")
    print(f"   (Use isso como baseline para comparar queries reais)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Teste do Mycelial Reasoning")
    
    parser.add_argument(
        '--state-path',
        type=str,
        default='./data/mycelial_state.pkl',
        help='Caminho para o estado da rede'
    )
    
    parser.add_argument(
        '--test-codes',
        type=str,
        default=None,
        help='Códigos para testar propagação. Formato: "10,20,30,40"'
    )
    
    args = parser.parse_args()
    
    # Carregar estado
    state = load_mycelial_state(args.state_path)
    
    # Análises
    print("\n🔍 Analisando saúde da rede...")
    health = analyze_network_health(state)
    
    print("🔍 Encontrando conexões mais fortes...")
    strongest = find_strongest_connections(state, top_k=20)
    
    print("🔍 Identificando hubs...")
    hubs = find_hub_codes(state, top_k=5)
    
    print("🔍 Calculando baseline aleatório...")
    random_baseline = compare_with_random(state, num_samples=100)
    
    # Relatório
    print_report(health, strongest, hubs, random_baseline)
    
    # Testes de propagação
    if args.test_codes:
        codes = [int(c) for c in args.test_codes.split(',')]
        test_cases = [{'name': 'Input do usuário', 'initial_codes': codes}]
    else:
        # Casos de teste padrão
        test_cases = [
            {'name': 'Padrão genérico (hubs)', 'initial_codes': [49, 255, 214, 255]},
            {'name': 'Padrão semi-raro', 'initial_codes': [10, 50, 100, 150]},
            {'name': 'Padrão raro', 'initial_codes': [3, 17, 89, 201]},
        ]
    
    print("\n" + "="*70)
    print("TESTES DE PROPAGAÇÃO")
    print("="*70)
    
    results = test_semantic_association(state, test_cases)
    
    # Resumo final
    print("\n" + "="*70)
    print("CONCLUSÃO")
    print("="*70)
    
    if health['total_active_connections'] > 1000:
        print("✅ Rede tem conexões suficientes")
    else:
        print("⚠️  Rede pode precisar de mais treinamento")
    
    if health['global_density'] < 0.05:
        print("✅ Densidade saudável (esparsa)")
    elif health['global_density'] < 0.20:
        print("⚠️  Densidade moderada")
    else:
        print("❌ Densidade muito alta (pode estar superconectada)")
    
    # Verificar distribuição de hubs (apenas informativo)
    print("ℹ️  Análise de Hubs (FineWeb):")
    for h in range(4):
        top_hub = hubs[h][0]
        print(f"   Head {h}: Top Hub {top_hub['code']} tem grau {top_hub['total_degree']:.0f}")
    
    print("\n")


if __name__ == '__main__':
    main()
