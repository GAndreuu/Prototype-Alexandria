#!/usr/bin/env python3
"""
Alexandria :: Teste Automatizado do MycelialReasoning (Functional)

Roda bateria completa de testes funcionais e reporta pass/fail.
Adaptado para MycelialReasoning v2.0 (Sparse Graph).

Uso:
    python test_mycelial_runner.py
    python test_mycelial_runner.py --verbose
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import shutil

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
    IMPORT_OK = True
except ImportError as e:
    IMPORT_OK = False
    IMPORT_ERROR = str(e)


# =============================================================================
# FRAMEWORK DE TESTE
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float


class TestRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def run(self, name: str, test_fn: Callable) -> bool:
        """Roda um teste e registra resultado."""
        start = time.time()
        try:
            passed, message = test_fn()
            duration = time.time() - start
            self.results.append(TestResult(name, passed, message, duration))
            
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} | {name}")
            if self.verbose or not passed:
                print(f"       {message}")
            
            return passed
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult(name, False, f"EXCEPTION: {e}", duration))
            print(f"[ERROR] | {name}")
            print(f"       {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def summary(self):
        """Imprime resumo final."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print("\n" + "=" * 60)
        print("RESUMO")
        print("=" * 60)
        print(f"Total:   {total}")
        print(f"Passou:  {passed}")
        print(f"Falhou:  {failed}")
        print(f"Taxa:    {100*passed/total:.1f}%")
        
        total_time = sum(r.duration for r in self.results)
        print(f"Tempo:   {total_time:.2f}s")
        
        if failed > 0:
            print("\nFalhas:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        print("=" * 60)
        
        return failed == 0


# =============================================================================
# TESTS
# =============================================================================

def test_import() -> Tuple[bool, str]:
    """Testa se o módulo importa corretamente."""
    if not IMPORT_OK:
        return False, f"Import falhou: {IMPORT_ERROR}"
    return True, "Módulo importado com sucesso"


def test_initialization() -> Tuple[bool, str]:
    """Testa inicialização básica (Sparse Graph)."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_init.pkl")
    m = MycelialReasoning(config)
    
    # Check structures
    stats = m.get_network_stats()
    
    checks = [
        isinstance(m.graph, dict),
        stats['active_nodes'] == 0,
        stats['active_edges'] == 0,
        m.total_observations == 0,
    ]
    
    if all(checks):
        return True, f"Graph init empty: OK"
    return False, f"Inicialização incorreta: {stats}"


def test_single_observation() -> Tuple[bool, str]:
    """Testa uma única observação e criação de nós."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_obs.pkl")
    m = MycelialReasoning(config)
    
    # Observe [h0=10, h1=20, h2=30, h3=40]
    m.observe([10, 20, 30, 40])
    
    stats = m.get_network_stats()
    
    # Expect 4 active nodes (one per head)
    # Expect edges between all pairs: 4 nodes * 3 = 12 directed edges (or 6 undirected)
    # Since graph is symmetric bidirectional in implementation:
    # (0,10)->(1,20), (1,20)->(0,10), etc.
    
    node_0_10 = (0, 10)
    node_1_20 = (1, 20)
    
    has_node = node_0_10 in m.graph
    has_connection = node_1_20 in m.graph[node_0_10]
    weight = m.graph[node_0_10][node_1_20]
    
    checks = [
        m.total_observations == 1,
        stats['active_nodes'] == 4,
        has_node,
        has_connection,
        weight > 0
    ]
    
    if all(checks):
        return True, f"Observação registrada, conexão 0:10->1:20 com peso {weight:.4f}"
    return False, f"Falha na observação. Stats: {stats}"


def test_hebbian_learning() -> Tuple[bool, str]:
    """Testa se observações repetidas fortalecem conexões."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_hebb.pkl")
    m = MycelialReasoning(config)
    
    # Observar mesmo padrão 10x
    indices = [10, 20, 30, 40]
    for _ in range(10):
        m.observe(indices)
    
    node_a = (0, 10)
    node_b = (1, 20)
    
    weight_strong = m.graph[node_a].get(node_b, 0.0)
    
    # Conexão inexistente
    weight_none = m.graph[node_a].get((2, 99), 0.0)
    
    # Expected weight ~= 10 * learning_rate (default 0.1) = 1.0 (approx)
    
    checks = [
        weight_strong > 0.5,
        weight_none == 0.0
    ]
    
    if all(checks):
        return True, f"Hebbian OK: Strong={weight_strong:.2f}, None={weight_none:.2f}"
    return False, f"Hebbian falhou: Strong={weight_strong:.2f}, Expected > 0.5"


def test_reasoning_pipeline() -> Tuple[bool, str]:
    """Testa propagação e raciocínio (completion)."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_reason.pkl")
    m = MycelialReasoning(config)
    
    # Treinar padrão forte: A->B
    # [10, 20, 30, 40]
    for _ in range(20):
        m.observe([10, 20, 30, 40])
        
    # Input parcial: [10, 0, 0, 0] (0 = vazio/ignorado logicamente se não tiver peso)
    # Assumindo que 0 não tem conexões fortes, a propagação de 10 deve ativar 20, 30, 40.
    
    # Nota: O sistema "reason" retorna indices refinados.
    # Se passarmos [10, 999, 999, 999] (ruído), ele deve sugerir [10, 20, 30, 40] se o sinal for forte.
    
    input_indices = [10, 0, 0, 0]
    result_indices, activations = m.reason(input_indices)
    
    # Esperamos que result_indices recupere [10, 20, 30, 40]
    # Head 0: 10 (mantido)
    # Head 1: deve ser 20
    # Head 2: deve ser 30
    # Head 3: deve ser 40
    
    matches = (
        result_indices[1] == 20 and
        result_indices[2] == 30 and
        result_indices[3] == 40
    )
    
    if matches:
        return True, f"Pattern Completion OK: {input_indices} -> {result_indices}"
    return False, f"Pattern Completion Failed: {input_indices} -> {result_indices} (Expected [10, 20, 30, 40])"


def test_hub_emergence() -> Tuple[bool, str]:
    """Testa identificação de hubs (nós muito conectados)."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_hubs.pkl")
    m = MycelialReasoning(config)
    
    # Nó (0, 10) aparece em vários contextos
    # Contexto 1
    for _ in range(5): m.observe([10, 20, 30, 40])
    # Contexto 2
    for _ in range(5): m.observe([10, 50, 60, 70])
    # Contexto 3
    for _ in range(5): m.observe([10, 80, 90, 100])
    
    hubs = m.get_hub_codes(top_k=5)
    
    if not hubs:
        return False, "Nenhum hub retornado"
        
    # Esperamos que (0, 10) seja o top hub
    top_hub = hubs[0]
    is_code_10 = (top_hub['head'] == 0 and top_hub['code'] == 10)
    
    if is_code_10:
        return True, f"Top hub identificado corretamente: Head 0, Code 10 (Degree: {top_hub['degree']})"
    
    return False, f"Top hub incorreto: {top_hub}"


def test_decay() -> Tuple[bool, str]:
    """Testa poda de conexões fracas."""
    config = MycelialConfig(
        save_path="/tmp/test_mycelial_decay.pkl",
        decay_rate=0.5, # Agressivo
        min_weight=0.6  # Threshold alto
    )
    m = MycelialReasoning(config)
    
    # Criar conexão com peso ~0.5 (5 obs * 0.1)
    for _ in range(5):
        m.observe([10, 20, 30, 40])
        
    stats_before = m.get_network_stats()
    
    # Aplicar decay
    # Peso 0.5 * 0.5 = 0.25 < 0.6 -> deve ser podado
    m.decay()
    
    stats_after = m.get_network_stats()
    
    if stats_after['active_edges'] < stats_before['active_edges']:
        return True, f"Decay funcionou: {stats_before['active_edges']} -> {stats_after['active_edges']} edges"
    
    return False, f"Decay falhou ou nada foi podado: {stats_after}"


def test_persistence() -> Tuple[bool, str]:
    """Testa salvar e carregar estado."""
    save_path = "/tmp/test_mycelial_persist.pkl"
    if Path(save_path).exists():
        Path(save_path).unlink()
        
    config = MycelialConfig(save_path=save_path)
    m1 = MycelialReasoning(config)
    
    # Criar estado
    m1.observe([10, 20, 30, 40])
    m1.save_state()
    
    # Carregar em nova instância
    m2 = MycelialReasoning(config) # Auto-loads in init
    
    stats1 = m1.get_network_stats()
    stats2 = m2.get_network_stats()
    
    checks = [
        stats1['total_observations'] == stats2['total_observations'],
        stats1['active_nodes'] == stats2['active_nodes'],
        stats1['active_edges'] == stats2['active_edges']
    ]
    
    if all(checks):
        return True, "Persistência OK (Stats idênticos)"
    return False, f"Falha na persistência. Original: {stats1}, Carregado: {stats2}"


def test_find_bridges_skip() -> Tuple[bool, str]:
    """Testa find_bridges (Feature removida/não impl. na v2)."""
    # Placeholder para manter relatório consistente se necessário
    return True, "SKIPPED: find_bridges not in v2 API"


def test_reset() -> Tuple[bool, str]:
    """Testa reset."""
    config = MycelialConfig(save_path="/tmp/test_mycelial_reset.pkl")
    m = MycelialReasoning(config)
    m.observe([10, 20, 30, 40])
    
    m.reset()
    
    if len(m.graph) == 0:
        return True, "Reset limpou grafo com sucesso"
    return False, "Grafo não vazio após reset"


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Testes Funcionais Mycelial v2")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALEXANDRIA :: TESTES MYCELIAL REASONING (V2 SPARSE)")
    print("=" * 60)
    print()
    
    runner = TestRunner(verbose=args.verbose)
    
    # Executar bateria
    runner.run("Import", test_import)
    if not IMPORT_OK:
        return 1
        
    runner.run("Initialization", test_initialization)
    runner.run("Single Observation", test_single_observation)
    runner.run("Hebbian Learning", test_hebbian_learning)
    runner.run("Reasoning Pipeline", test_reasoning_pipeline)
    runner.run("Hub Emergence", test_hub_emergence)
    runner.run("Decay System", test_decay)
    runner.run("Persistence", test_persistence)
    runner.run("Reset", test_reset)
    
    success = runner.summary()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
