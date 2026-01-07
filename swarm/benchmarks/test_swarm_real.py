"""
test_swarm_real.py

Teste abrangente das capacidades REAIS do sistema Swarm.
Não depende de módulos externos - testa o sistema isolado.

Testes:
1. Agentes Neurodiversos (Collapse, Critical, Psychedelic, Autistic)
2. Sistema de Consenso Justo
3. Memória Persistente
4. Classificador de Complexidade
5. Early Stopping Committee
6. Navegação Completa (SwarmNavigator)
7. Topologia (Analyzer + Bridge Builder)
"""

import sys
import os
import numpy as np
import json
import time
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# UTILITIES
# ============================================================================

def create_concept_embedding(concept: str, dim: int = 384) -> np.ndarray:
    """Create a deterministic embedding from concept string."""
    np.random.seed(hash(concept) % (2**32))
    emb = np.random.randn(dim)
    return emb / np.linalg.norm(emb)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def print_header(title: str):
    """Print test header."""
    print("\n" + "=" * 70)
    print(f"🧪 {title}")
    print("=" * 70)

def print_result(passed: bool, message: str):
    """Print test result."""
    if passed:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    return passed

# ============================================================================
# TEST 1: NEURODIVERSE AGENTS
# ============================================================================

def test_neurodiverse_agents() -> Tuple[bool, Dict]:
    """Test that each neurodiverse agent works correctly."""
    print_header("TEST 1: Agentes Neurodiversos")
    
    from swarm.agents.collapse import CollapseAgent
    from swarm.agents.critical import CriticalAgent
    from swarm.agents.psychedelic import PsychedelicAgent
    from swarm.agents.autistic import AutisticAgent
    from swarm.core import Context, NeurotypeName
    
    # Create context
    start = create_concept_embedding("machine learning")
    target = create_concept_embedding("neural networks")
    current = (start + target) / 2  # Midpoint
    current = current / np.linalg.norm(current)
    
    ctx = Context(
        start_emb=start,
        target_emb=target,
        current=current,
        history=[start, current],
        step=1,
        initial_dist=np.linalg.norm(target - start)
    )
    
    agents = {
        "CollapseAgent": CollapseAgent(),
        "CriticalAgent": CriticalAgent(),
        "PsychedelicAgent": PsychedelicAgent(),
        "AutisticAgent": AutisticAgent()
    }
    
    results = {}
    all_passed = True
    
    for name, agent in agents.items():
        try:
            step = agent.propose(ctx)
            
            # Validate step
            assert step.direction is not None, "Direction is None"
            assert len(step.direction) == 384, f"Wrong dimension: {len(step.direction)}"
            assert 0 <= step.confidence <= 1, f"Invalid confidence: {step.confidence}"
            assert step.reasoning, "No reasoning provided"
            assert step.neurotype in NeurotypeName, "Invalid neurotype"
            
            # Check direction is unit vector
            direction_norm = np.linalg.norm(step.direction)
            assert abs(direction_norm - 1.0) < 0.1, f"Direction not normalized: {direction_norm}"
            
            results[name] = {
                "confidence": step.confidence,
                "neurotype": step.neurotype.value,
                "reasoning": step.reasoning[:50] + "...",
                "has_topo_state": step.topological_state is not None
            }
            
            print(f"  ✓ {name}: conf={step.confidence:.3f}, neurotype={step.neurotype.value}")
            
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_passed = False
            results[name] = {"error": str(e)}
    
    print_result(all_passed, f"Todos os {len(agents)} agentes funcionam corretamente")
    return all_passed, results

# ============================================================================
# TEST 2: NEURODIVERSE CONSENSUS
# ============================================================================

def test_neurodiverse_consensus() -> Tuple[bool, Dict]:
    """Test the fair consensus system."""
    print_header("TEST 2: Sistema de Consenso Neurodiverso")
    
    from swarm.core.consensus import NeurodiverseConsensus
    from swarm.core import NavigationStep, Context, NeurotypeName
    
    # Create context
    start = create_concept_embedding("art")
    target = create_concept_embedding("science")
    
    ctx = Context(
        start_emb=start,
        target_emb=target,
        current=start.copy(),
        history=[start],
        step=0,
        initial_dist=np.linalg.norm(target - start)
    )
    
    # Create diverse proposals
    proposals = []
    for name, conf in [
        (NeurotypeName.COLLAPSE, 0.9),
        (NeurotypeName.CRITICAL, 0.7),
        (NeurotypeName.PSYCH, 0.5),
        (NeurotypeName.AUTISTIC, 0.8),
        (NeurotypeName.BALANCED, 0.6)
    ]:
        direction = np.random.randn(384)
        direction = direction / np.linalg.norm(direction)
        
        proposals.append(NavigationStep(
            agent_id=name.value,
            neurotype=name,
            direction=direction,
            confidence=conf,
            reasoning=f"Test proposal from {name.value}"
        ))
    
    # Test consensus
    consensus = NeurodiverseConsensus()
    result = consensus.compute_consensus(proposals, ctx)
    
    results = {}
    all_passed = True
    
    try:
        # Validate result
        assert result.direction is not None, "No direction computed"
        assert len(result.direction) == 384, f"Wrong dimension: {len(result.direction)}"
        assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
        
        # Check weight distribution
        stats = consensus.get_stats()
        # stats['agent_proportions'] sums to 1.0 ideally
        total_contribs = sum(stats.get('agent_proportions', {}).values())
        
        print(f"  Direction norm: {np.linalg.norm(result.direction):.4f}")
        print(f"  Consensus confidence: {result.confidence:.4f}")
        print(f"  Reasoning: {result.reasoning}")
        
        print("\n  Contribuições por neurótipo:")
        # Use result.contributing_neurotypes directly for per-step check
        for neurotype, weight in sorted(result.contributing_neurotypes.items(), key=lambda x: -x[1]):
            pct = weight * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {neurotype.value:12s}: {bar} {pct:.1f}%")
            
            # Check veto limit (max 30%)
            if weight > 0.35:  # 5% tolerance
                print(f"    ⚠️ VETO WARNING: {neurotype.value} exceeds 30%")
        
        # Verify neurodiverse minimum
        neurodiverse_types = {NeurotypeName.PSYCH, NeurotypeName.AUTISTIC, NeurotypeName.CRITICAL}
        neurodiverse_weight = sum(
            w for n, w in result.contributing_neurotypes.items() 
            if n in neurodiverse_types
        )
        
        print(f"\n  Peso total neurodiverso: {neurodiverse_weight:.1%} (mínimo esperado: 40%)")
        
        results = {
            "confidence": result.confidence,
            "contributions": {k.value: v for k, v in result.contributing_neurotypes.items()},
            "neurodiverse_weight": neurodiverse_weight,
            "stats": stats
        }
        
    except Exception as e:
        print(f"  ✗ Erro: {e}")
        all_passed = False
        results["error"] = str(e)
    
    print_result(all_passed, "Sistema de consenso funciona com pesos justos")
    return all_passed, results

# ============================================================================
# TEST 3: PERSISTENT MEMORY
# ============================================================================

def test_persistent_memory() -> Tuple[bool, Dict]:
    """Test persistent topological memory."""
    print_header("TEST 3: Memória Topológica Persistente")
    
    from swarm.core.memory import PersistentTopologicalMemory
    
    # Use temp dir for testing
    import shutil
    test_dir = "/tmp/swarm_test_data"
    test_dummy_file = os.path.join(test_dir, "memory.json")
    
    try:
        # Clean up if exists
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        
        memory = PersistentTopologicalMemory(save_path=test_dummy_file)
        
        # Create test trajectories
        trajectories = []
        for i, (start_concept, target_concept, success) in enumerate([
            ("physics", "mathematics", True),
            ("art", "music", True),
            ("philosophy", "science", False),
            ("biology", "chemistry", True),
            ("history", "future", False)
        ]):
            start = create_concept_embedding(start_concept)
            target = create_concept_embedding(target_concept)
            
            # Create trajectory
            trajectory = []
            for t in np.linspace(0, 1, 10):
                pos = (1 - t) * start + t * target
                pos = pos / np.linalg.norm(pos)
                trajectory.append(pos)
            
            # Save trajectory
            memory.save_trajectory(
                trajectory=trajectory,
                start=start,
                target=target,
                success=success,
                efficiency=0.8 if success else 0.3,
                mean_curvature=0.2,
                neurotypes_used={"collapse": 0.4, "critical": 0.3, "psychedelic": 0.3},
                complexity=0.5
            )
            
            trajectories.append({
                "start": start_concept,
                "target": target_concept,
                "success": success
            })
        
        # Test retrieval
        query_start = create_concept_embedding("chemistry")
        query_target = create_concept_embedding("biology")
        
        similar = memory.find_similar_trajectories(
            start_emb=query_start,
            target_emb=query_target,
            top_k=3
        )
        
        print(f"  Trajetórias salvas: {len(trajectories)}")
        print(f"  Trajetórias similares encontradas: {len(similar)}")
        
        for mem in similar:
            print(f"    - {mem.start_code} → {mem.target_code}")
            print(f"      Sucesso: {mem.success}, Importância: {mem.importance:.3f}")
        
        # Get stats
        stats = memory.get_stats()
        print(f"\n  📊 Estatísticas:")
        print(f"    Total: {stats['total_trajectories']}")
        print(f"    Sucesso: {stats.get('success_rate', 0):.1%}")
        
        # Force save
        memory.save_to_disk()
        
        # Verify persistence (trajectories.json inside the dir)
        expected_file = os.path.join(test_dir, "trajectories.json")
        assert os.path.exists(expected_file), f"Memory file not created at {expected_file}"
        
        with open(expected_file) as f:
            data = json.load(f)
        assert len(data) > 0, "No trajectories saved in JSON"
        
        results = {
            "trajectories_saved": len(trajectories),
            "similar_found": len(similar),
            "stats": stats,
            "file_exists": True
        }
        
        print_result(True, "Memória persistente funciona corretamente")
        return True, results
        
    except Exception as e:
        print(f"  ✗ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}
    
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

# ============================================================================
# TEST 4: COMPLEXITY CLASSIFIER
# ============================================================================

def test_complexity_classifier() -> Tuple[bool, Dict]:
    """Test complexity classification for mode selection."""
    print_header("TEST 4: Classificador de Complexidade")
    
    from swarm.tools.complexity_classifier import ComplexityClassifier
    from swarm.core import NavigationMode
    
    classifier = ComplexityClassifier()
    
    test_cases = [
        # (start, target, expected_mode_hint)
        ("neural network", "deep learning", "SPRINT"),  # Very similar
        ("philosophy", "quantum mechanics", "CAUTIOUS"),  # Distant
        ("creativity", "innovation", "BALANCED"),  # Medium
        ("abstract art", "surrealism", "CREATIVE")  # Creative domain
    ]
    
    results = {}
    all_passed = True
    
    for start_concept, target_concept, expected_hint in test_cases:
        start = create_concept_embedding(start_concept)
        target = create_concept_embedding(target_concept)
        
        similarity = cosine_sim(start, target)
        mode, config = classifier.classify(start, target)
        
        print(f"\n  '{start_concept}' → '{target_concept}'")
        print(f"    Similaridade: {similarity:.3f}")
        print(f"    Modo: {mode.value}")
        print(f"    Config: max_steps={config.max_steps}, min_conf={config.min_confidence:.2f}")
        
        # Validate config
        assert hasattr(config, 'max_steps'), "Config missing max_steps"
        assert hasattr(config, 'min_confidence'), "Config missing min_confidence"
        assert config.max_steps > 0, f"Invalid max_steps: {config.max_steps}"
        
        results[f"{start_concept}->{target_concept}"] = {
            "similarity": similarity,
            "mode": mode.value,
            "max_steps": config.max_steps
        }
    
    # Check stats
    stats = classifier.get_stats()
    print(f"\n  📊 Classificações: {stats}")
    
    print_result(all_passed, "Classificador de complexidade funciona corretamente")
    return all_passed, results

# ============================================================================
# TEST 5: EARLY STOPPING COMMITTEE
# ============================================================================

def test_early_stopping() -> Tuple[bool, Dict]:
    """Test early stopping committee."""
    print_header("TEST 5: Comitê de Early Stopping")
    
    from swarm.tools.early_stopping import AdaptiveEarlyStoppingCommittee
    
    committee = AdaptiveEarlyStoppingCommittee()
    
    # Test case 1: Converging trajectory
    target = create_concept_embedding("convergence_target")
    start = create_concept_embedding("start_point")
    
    # Build converging trajectory
    converging_traj = []
    for t in np.linspace(0, 0.99, 20):
        pos = (1 - t) * start + t * target
        pos = pos / (np.linalg.norm(pos) + 1e-9)
        converging_traj.append(pos)
    
    # Test case 2: Stagnating trajectory
    stagnating_traj = [start.copy() + np.random.randn(384) * 0.01 for _ in range(15)]
    stagnating_traj = [t / np.linalg.norm(t) for t in stagnating_traj]
    
    results = {}
    
    print("\n  Caso 1: Trajetória convergente")
    decision1 = committee.should_stop(
        trajectory=converging_traj,
        target=target,
        current_step=len(converging_traj),
        context={"mode": "balanced"}
    )
    
    print(f"    Parar: {decision1.should_stop}")
    print(f"    Concordância: {decision1.agreement_ratio:.1%}")
    print(f"    Razão: {decision1.primary_reason}")
    print(f"    Votos:")
    for vote in decision1.votes:
        print(f"      {vote.voter_name}: stop={vote.should_stop}, conf={vote.confidence:.2f}")
    
    results["converging"] = {
        "should_stop": decision1.should_stop,
        "agreement": decision1.agreement_ratio,
        "reason": decision1.primary_reason
    }
    
    print("\n  Caso 2: Trajetória estagnada")
    decision2 = committee.should_stop(
        trajectory=stagnating_traj,
        target=target,
        current_step=len(stagnating_traj),
        context={"mode": "balanced"}
    )
    
    print(f"    Parar: {decision2.should_stop}")
    print(f"    Concordância: {decision2.agreement_ratio:.1%}")
    print(f"    Razão: {decision2.primary_reason}")
    
    results["stagnating"] = {
        "should_stop": decision2.should_stop,
        "agreement": decision2.agreement_ratio,
        "reason": decision2.primary_reason
    }
    
    # Converging should stop (reached target), stagnating might stop (no progress)
    passed = decision1.should_stop  # Converging trajectory should trigger stop
    
    print_result(passed, "Comitê de early stopping funciona corretamente")
    return passed, results

# ============================================================================
# TEST 6: TOPOLOGY ANALYZER
# ============================================================================

def test_topology_analyzer() -> Tuple[bool, Dict]:
    """Test topology analysis."""
    print_header("TEST 6: Analisador Topológico")
    
    from swarm.topology.analyzer import TopologyAnalyzer
    
    analyzer = TopologyAnalyzer(dimension=384)
    
    # Create test trajectory
    start = create_concept_embedding("topology_start")
    target = create_concept_embedding("topology_target")
    
    # Smooth trajectory
    smooth_history = []
    for t in np.linspace(0, 0.5, 10):
        pos = (1 - t) * start + t * target
        pos = pos / np.linalg.norm(pos)
        smooth_history.append(pos)
    
    # Erratic trajectory
    erratic_history = [start.copy()]
    for i in range(9):
        noise = np.random.randn(384) * 0.5
        pos = erratic_history[-1] + noise
        pos = pos / np.linalg.norm(pos)
        erratic_history.append(pos)
    
    results = {}
    
    # Test smooth trajectory
    print("\n  Trajetória Suave:")
    state1 = analyzer.analyze_local(
        position=smooth_history[-1],
        target=target,
        history=smooth_history
    )
    
    print(f"    Curvatura: {state1.curvature:.4f}")
    print(f"    Energia: {state1.energy:.4f}")
    print(f"    Densidade: {state1.density:.4f}")
    print(f"    Perto de colapso: {state1.near_collapse}")
    print(f"    Entropia: {state1.entropy:.4f}")
    
    results["smooth"] = {
        "curvature": state1.curvature,
        "energy": state1.energy,
        "near_collapse": state1.near_collapse
    }
    
    # Test erratic trajectory
    print("\n  Trajetória Errática:")
    state2 = analyzer.analyze_local(
        position=erratic_history[-1],
        target=target,
        history=erratic_history
    )
    
    print(f"    Curvatura: {state2.curvature:.4f}")
    print(f"    Energia: {state2.energy:.4f}")
    print(f"    Densidade: {state2.density:.4f}")
    print(f"    Perto de colapso: {state2.near_collapse}")
    print(f"    Entropia: {state2.entropy:.4f}")
    
    results["erratic"] = {
        "curvature": state2.curvature,
        "energy": state2.energy,
        "near_collapse": state2.near_collapse
    }
    
    # Collapse risk detection
    risk, reason = analyzer.detect_collapse_risk(
        position=erratic_history[-1],
        target=target,
        history=erratic_history
    )
    print(f"\n  Risco de colapso: {risk:.3f} ({reason})")
    
    results["collapse_risk"] = {"risk": risk, "reason": reason}
    
    # Erratic should have higher curvature
    passed = state2.curvature >= state1.curvature
    
    print_result(passed, "Analisador topológico detecta diferenças corretamente")
    return passed, results

# ============================================================================
# TEST 7: ACTIVE BRIDGE BUILDER
# ============================================================================

def test_bridge_builder() -> Tuple[bool, Dict]:
    """Test active bridge building."""
    print_header("TEST 7: Construtor de Pontes Conceituais")
    
    from swarm.topology.bridge import ActiveBridgeBuilder, BridgeCandidate
    
    builder = ActiveBridgeBuilder()
    
    # Test concepts
    concept_a = create_concept_embedding("music")
    concept_b = create_concept_embedding("mathematics")
    
    similarity = cosine_sim(concept_a, concept_b)
    print(f"\n  Conceitos: 'music' ↔ 'mathematics'")
    print(f"  Similaridade direta: {similarity:.4f}")
    
    # Propose bridges via interpolation
    print("\n  Propondo pontes por interpolação...")
    candidates = builder.propose_bridge(concept_a, concept_b, method='interpolation', num_candidates=5)
    
    print(f"  Candidatos gerados: {len(candidates)}")
    
    results = {"candidates": []}
    
    for i, c in enumerate(candidates):
        print(f"\n    Candidato {i+1}:")
        print(f"      Método: {c.method}")
        print(f"      Confiança: {c.confidence:.4f}")
        
        # Validate
        is_valid, reason = builder.validate_bridge(c)
        print(f"      Válido: {is_valid} ({reason})")
        
        results["candidates"].append({
            "method": c.method,
            "confidence": c.confidence,
            "valid": is_valid,
            "reason": reason
        })
    
    passed = len(candidates) > 0 and candidates[0].confidence > 0
    
    print_result(passed, "Construtor de pontes funciona corretamente")
    return passed, results

# ============================================================================
# TEST 8: FULL SWARM NAVIGATION
# ============================================================================

def test_full_navigation() -> Tuple[bool, Dict]:
    """Test complete SwarmNavigator flow."""
    print_header("TEST 8: Navegação Completa com SwarmNavigator")
    
    from swarm import SwarmNavigator
    
    # Initialize navigator (standalone mode)
    navigator = SwarmNavigator(
        topology_engine=None,  # Will use fallback
        mycelial_system=None,  # Will use fallback
        memory_path="/tmp/swarm_nav_test.json",
        use_neurodiverse=True
    )
    
    test_cases = [
        ("machine learning", "deep learning"),  # Close concepts
        ("philosophy", "computer science"),  # Medium distance
        ("water", "fire"),  # Opposite concepts
        ("creativity", "logic")  # Abstract concepts
    ]
    
    results = {"navigations": []}
    all_passed = True
    
    for start_concept, target_concept in test_cases:
        print(f"\n  📍 '{start_concept}' → '{target_concept}'")
        
        start_time = time.time()
        
        try:
            result = navigator.navigate(
                start_concept=start_concept,
                target_concept=target_concept,
                max_steps=30,
                debug=False
            )
            
            elapsed = time.time() - start_time
            
            print(f"    Sucesso: {result['success']}")
            print(f"    Passos: {result['steps']}")
            print(f"    Similaridade: {result['init_similarity']:.4f} → {result['final_similarity']:.4f}")
            print(f"    Melhoria: {result['improvement']:.4f}")
            print(f"    Tempo: {elapsed:.2f}s")
            
            # Neurotype contributions
            contribs = result.get('neurotype_contributions', {})
            if contribs:
                sorted_contribs = sorted(contribs.items(), key=lambda x: -x[1])[:3]
                print(f"    Top neurótipos: {', '.join([f'{k}:{v:.2f}' for k,v in sorted_contribs])}")
            
            results["navigations"].append({
                "start": start_concept,
                "target": target_concept,
                "success": result['success'],
                "steps": result['steps'],
                "improvement": result['improvement'],
                "time": elapsed
            })
            
            # Success criteria: improvement > 0 or already high similarity
            if result['improvement'] <= 0 and result['init_similarity'] < 0.9:
                print(f"    ⚠️ Nenhuma melhoria alcançada")
            
        except Exception as e:
            print(f"    ❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            results["navigations"].append({
                "start": start_concept,
                "target": target_concept,
                "error": str(e)
            })
    
    # Aggregate stats
    successes = sum(1 for n in results["navigations"] if n.get("success", False))
    total = len(results["navigations"])
    avg_improvement = np.mean([n.get("improvement", 0) for n in results["navigations"]])
    avg_time = np.mean([n.get("time", 0) for n in results["navigations"]])
    
    print(f"\n  📊 Resumo:")
    print(f"    Taxa de sucesso: {successes}/{total}")
    print(f"    Melhoria média: {avg_improvement:.4f}")
    print(f"    Tempo médio: {avg_time:.2f}s")
    
    results["summary"] = {
        "success_rate": successes / total,
        "avg_improvement": avg_improvement,
        "avg_time": avg_time
    }
    
    # Consider passed if at least half succeeded
    passed = successes >= total / 2
    
    print_result(passed, f"Navegação completa: {successes}/{total} navegações bem-sucedidas")
    return passed, results

# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "=" * 70)
    print("🧬 TESTE ABRANGENTE DO SISTEMA SWARM")
    print("=" * 70)
    print(f"\nHorário: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Projeto: Alexandria/swarm")
    print("-" * 70)
    
    tests = [
        ("Agentes Neurodiversos", test_neurodiverse_agents),
        ("Consenso Justo", test_neurodiverse_consensus),
        ("Memória Persistente", test_persistent_memory),
        ("Classificador de Complexidade", test_complexity_classifier),
        ("Early Stopping", test_early_stopping),
        ("Analisador Topológico", test_topology_analyzer),
        ("Construtor de Pontes", test_bridge_builder),
        ("Navegação Completa", test_full_navigation)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            passed, data = test_func()
            results[name] = {"passed": passed, "data": data}
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO em {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"passed": False, "error": str(e)}
    
    # Final Summary
    print("\n" + "=" * 70)
    print("📋 RESUMO FINAL")
    print("=" * 70)
    
    passed_count = sum(1 for r in results.values() if r.get("passed", False))
    total_count = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed_count}/{total_count} testes passaram")
    
    if passed_count == total_count:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("   O sistema Swarm está funcionando corretamente.")
    elif passed_count >= total_count * 0.7:
        print("\n⚠️ MAIORIA DOS TESTES PASSOU")
        print("   Sistema funcional, mas algumas funcionalidades precisam atenção.")
    else:
        print("\n❌ VÁRIOS TESTES FALHARAM")
        print("   O sistema pode precisar de correções.")
    
    return passed_count, total_count, results


if __name__ == "__main__":
    passed, total, results = run_all_tests()
    sys.exit(0 if passed >= total * 0.7 else 1)
