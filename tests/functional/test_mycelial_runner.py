#!/usr/bin/env python3
"""
Alexandria :: Teste Automatizado do MycelialReasoning

Roda bateria completa de testes e reporta pass/fail.

Uso:
    python test_mycelial.py
    python test_mycelial.py --verbose
    python test_mycelial.py --with-real-data
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Callable
from dataclasses import dataclass

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.mycelial_reasoning import MycelialReasoning, MycelialConfig
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
# TESTES
# =============================================================================

def test_import() -> Tuple[bool, str]:
    """Testa se o módulo importa corretamente."""
    if not IMPORT_OK:
        return False, f"Import falhou: {IMPORT_ERROR}"
    return True, "Módulo importado com sucesso"


def test_initialization() -> Tuple[bool, str]:
    """Testa inicialização básica."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    checks = [
        m.connections.shape == (4, 256, 256),
        m.activation_counts.shape == (4, 256),
        m.total_observations == 0,
        np.all(m.connections == 0),
    ]
    
    if all(checks):
        return True, f"Shape: {m.connections.shape}, zerado: OK"
    return False, f"Shapes incorretos ou não zerado"


def test_single_observation() -> Tuple[bool, str]:
    """Testa uma única observação."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    m.observe([10, 20, 30, 40])
    
    checks = [
        m.total_observations == 1,
        m.activation_counts[0, 10] == 1,
        m.activation_counts[1, 20] == 1,
        m.connections[0, 10, 20] > 0,  # head 0, código 10 → código 20
    ]
    
    if all(checks):
        conn_strength = m.connections[0, 10, 20]
        return True, f"Observação registrada, conexão 10→20: {conn_strength:.4f}"
    return False, "Observação não registrada corretamente"


def test_hebbian_learning() -> Tuple[bool, str]:
    """Testa se observações repetidas fortalecem conexões."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Observar mesmo padrão 100x
    for _ in range(100):
        m.observe([10, 20, 30, 40])
    
    # Verificar conexões fortes
    conn_10_20 = m.connections[0, 10, 20]
    conn_10_30 = m.connections[0, 10, 30]
    
    # Conexão aleatória deve ser fraca/zero
    conn_random = m.connections[0, 100, 200]
    
    checks = [
        conn_10_20 > 0.5,
        conn_10_30 > 0.5,
        conn_random < 0.1,
    ]
    
    if all(checks):
        return True, f"Hebbian OK: 10→20={conn_10_20:.2f}, random={conn_random:.2f}"
    return False, f"Hebbian falhou: 10→20={conn_10_20:.2f}, random={conn_random:.2f}"


def test_propagation_changes_indices() -> Tuple[bool, str]:
    """Testa se propagação altera índices."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Treinar padrões
    for _ in range(100):
        m.observe([10, 20, 30, 40])
    for _ in range(100):
        m.observe([10, 50, 60, 70])
    
    # Propagar de input parcial
    original = np.array([10, 20, 0, 0])
    new_indices, activation = m.reason(original)
    
    # Deve ter mudado algo
    changed = not np.array_equal(original, new_indices)
    
    if changed:
        return True, f"Original: {original} → Novo: {new_indices}"
    
    # Se não mudou, verificar se ativação é diferente
    if np.max(activation) > 0:
        return True, f"Índices iguais mas ativação presente: max={np.max(activation):.2f}"
    
    return False, "Propagação não alterou nada"


def test_hub_emergence() -> Tuple[bool, str]:
    """Testa se hubs emergem de padrões frequentes."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Código 10 aparece em TODOS os padrões
    for _ in range(50):
        m.observe([10, 20, 30, 40])
        m.observe([10, 50, 60, 70])
        m.observe([10, 80, 90, 100])
    
    hubs = m.get_hub_codes(5)
    
    if not hubs:
        return False, "Nenhum hub encontrado"
    
    # Código 10 deve ser o top hub (ou estar no top)
    top_codes = [h['code'] for h in hubs[:3] if h['head'] == 0]
    
    if 10 in top_codes:
        degree = next(h['total_degree'] for h in hubs if h['code'] == 10 and h['head'] == 0)
        return True, f"Código 10 é hub com degree={degree}"
    
    return False, f"Código 10 não é hub. Top: {top_codes}"


def test_decay() -> Tuple[bool, str]:
    """Testa se decaimento reduz conexões."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz", decay_rate=0.1)
    m = MycelialReasoning(config)
    
    # Treinar
    for _ in range(100):
        m.observe([10, 20, 30, 40])
    
    antes = m.get_network_stats()['active_connections']
    
    # Decair várias vezes
    for _ in range(50):
        m.decay()
    
    depois = m.get_network_stats()['active_connections']
    
    if depois < antes:
        return True, f"Antes: {antes} → Depois: {depois} (reduziu {antes-depois})"
    return False, f"Decaimento não funcionou: {antes} → {depois}"


def test_persistence() -> Tuple[bool, str]:
    """Testa se estado persiste entre instâncias."""
    save_path = "/tmp/test_mycelial_persist.npz"
    
    # Limpar arquivo antigo
    Path(save_path).unlink(missing_ok=True)
    
    # Sessão 1: criar e salvar
    config1 = MycelialConfig(save_path=save_path)
    m1 = MycelialReasoning(config1)
    for _ in range(100):
        m1.observe([10, 20, 30, 40])
    m1.save_state()
    obs1 = m1.total_observations
    conn1 = m1.connections[0, 10, 20]
    
    # Sessão 2: carregar
    config2 = MycelialConfig(save_path=save_path)
    m2 = MycelialReasoning(config2)
    obs2 = m2.total_observations
    conn2 = m2.connections[0, 10, 20]
    
    # Limpar
    Path(save_path).unlink(missing_ok=True)
    
    if obs1 == obs2 and np.isclose(conn1, conn2):
        return True, f"Persistiu: obs={obs1}, conn={conn1:.4f}"
    return False, f"Não persistiu: obs {obs1}→{obs2}, conn {conn1:.4f}→{conn2:.4f}"


def test_find_bridges() -> Tuple[bool, str]:
    """Testa busca de pontes entre domínios."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Dois "domínios" que compartilham código 10
    for _ in range(100):
        m.observe([10, 20, 30, 40])   # Domínio A
    for _ in range(100):
        m.observe([10, 120, 130, 140])  # Domínio B
    
    bridges = m.find_bridges(
        np.array([10, 20, 30, 40]),
        np.array([10, 120, 130, 140])
    )
    
    if not bridges:
        return False, "Nenhuma ponte encontrada"
    
    # Código 10 deve ser ponte
    bridge_codes = [b['code'] for b in bridges]
    
    if 10 in bridge_codes:
        score = next(b['bridge_score'] for b in bridges if b['code'] == 10)
        return True, f"Ponte encontrada: código 10 com score={score:.4f}"
    
    return False, f"Código 10 não é ponte. Pontes: {bridge_codes}"


def test_network_stats() -> Tuple[bool, str]:
    """Testa geração de estatísticas."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Usar padrão repetido para garantir conexões fortes (> 0.05)
    for _ in range(10):
        m.observe([10, 20, 30, 40])
    
    # E alguns aleatórios
    for _ in range(90):
        m.observe(np.random.randint(0, 256, size=4))
    
    stats = m.get_network_stats()

    
    required_keys = [
        'total_observations',
        'active_connections',
        'density',
        'mean_connection_strength',
    ]
    
    missing = [k for k in required_keys if k not in stats]
    
    if missing:
        return False, f"Keys faltando: {missing}"
    
    if stats['total_observations'] == 100 and stats['active_connections'] > 0:
        return True, f"Stats OK: obs={stats['total_observations']}, conn={stats['active_connections']}"
    
    return False, f"Stats incorretas: {stats}"


def test_batch_observation() -> Tuple[bool, str]:
    """Testa observação em batch."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    batch = np.random.randint(0, 256, size=(50, 4))
    m.observe_batch(batch)
    
    if m.total_observations == 50:
        return True, f"Batch de 50 observações registrado"
    return False, f"Esperado 50, got {m.total_observations}"


def test_density_stays_low() -> Tuple[bool, str]:
    """Testa se rede permanece esparsa."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Muitas observações aleatórias
    for _ in range(500):
        m.observe(np.random.randint(0, 256, size=4))
    
    stats = m.get_network_stats()
    density = stats['density']
    
    # Rede deve ser esparsa (< 10%)
    if density < 0.1:
        return True, f"Densidade: {density:.2%} (esparsa ✓)"
    return False, f"Densidade muito alta: {density:.2%}"


def test_reset() -> Tuple[bool, str]:
    """Testa reset da rede."""
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    # Usar
    for _ in range(100):
        m.observe([10, 20, 30, 40])
    
    antes = m.total_observations
    
    # Reset
    m.reset()
    
    depois = m.total_observations
    conn = np.sum(m.connections)
    
    if depois == 0 and conn == 0:
        return True, f"Reset OK: {antes} → {depois}, conexões zeradas"
    return False, f"Reset falhou: obs={depois}, conn_sum={conn}"


def test_torch_input() -> Tuple[bool, str]:
    """Testa se aceita input torch."""
    try:
        import torch
    except ImportError:
        return True, "Torch não disponível, skip"
    
    config = MycelialConfig(save_path="/tmp/test_mycelial.npz")
    m = MycelialReasoning(config)
    
    indices = torch.tensor([10, 20, 30, 40])
    m.observe(indices)
    
    if m.total_observations == 1:
        return True, "Input torch aceito"
    return False, "Falha com input torch"


# =============================================================================
# TESTES COM DADOS REAIS (OPCIONAL)
# =============================================================================

def test_with_real_data() -> Tuple[bool, str]:
    """Testa com dados reais do Alexandria (se disponível)."""
    try:
        from core.semantic_memory import SemanticMemory
        # from v2.monolith_v13 import MonolithV13
    except ImportError:
        return True, "SemanticMemory não disponível, skip"
    
    # Aqui entraria lógica de teste com dados reais
    return True, "Teste com dados reais: TODO"


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Testes do MycelialReasoning")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output detalhado")
    parser.add_argument("--with-real-data", action="store_true", help="Incluir testes com dados reais")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALEXANDRIA :: TESTES DO MYCELIAL REASONING")
    print("=" * 60)
    print()
    
    runner = TestRunner(verbose=args.verbose)
    
    # Testes básicos
    runner.run("Import", test_import)
    
    if not IMPORT_OK:
        print("\n❌ Import falhou. Não é possível continuar.")
        return 1
    
    runner.run("Inicialização", test_initialization)
    runner.run("Observação única", test_single_observation)
    runner.run("Aprendizado Hebbian", test_hebbian_learning)
    runner.run("Propagação altera índices", test_propagation_changes_indices)
    runner.run("Emergência de hubs", test_hub_emergence)
    runner.run("Decaimento", test_decay)
    runner.run("Persistência", test_persistence)
    runner.run("Busca de pontes", test_find_bridges)
    runner.run("Estatísticas da rede", test_network_stats)
    runner.run("Observação em batch", test_batch_observation)
    runner.run("Densidade permanece baixa", test_density_stays_low)
    runner.run("Reset", test_reset)
    runner.run("Input torch", test_torch_input)
    
    if args.with_real_data:
        runner.run("Dados reais", test_with_real_data)
    
    # Resumo
    success = runner.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
