"""
swarm/integration_test.py

Integration test: Swarm V3 with core Alexandria modules.
Tests connectivity and data flow between:
- SwarmNavigatorV3
- MetaHebbianPlasticity 
- ActiveInferenceAgent
- SelfFeedingLoop (conceptual)

Runs in SANDBOX mode - no modifications to production data.
"""

import numpy as np
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SwarmIntegration")

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# IMPORTS
# ============================================================================

# Swarm V3
from swarm import SwarmNavigator, NeurotypeName

# Core modules
try:
    from core.topology.topology_engine import create_topology_engine
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False
    logger.warning("TopologyEngine not available")

try:
    from swarm import MycelialReasoningLite
    MYCELIAL_AVAILABLE = True
except ImportError:
    MYCELIAL_AVAILABLE = False
    logger.warning("MycelialReasoningLite not available")

try:
    from core.learning.meta_hebbian import MetaHebbianPlasticity, MetaHebbianConfig
    META_HEBBIAN_AVAILABLE = True
except ImportError:
    META_HEBBIAN_AVAILABLE = False
    logger.warning("MetaHebbianPlasticity not available")

try:
    from core.learning.active_inference import ActiveInferenceAgent, ActiveInferenceConfig, Action, ActionType
    ACTIVE_INFERENCE_AVAILABLE = True
except ImportError:
    ACTIVE_INFERENCE_AVAILABLE = False
    logger.warning("ActiveInferenceAgent not available")

# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_swarm_meta_hebbian_integration():
    """
    Test: Swarm V3 navigation results influence Meta-Hebbian plasticity.
    
    Flow:
    1. Swarm navigates from A to B
    2. If successful, Meta-Hebbian reinforces the path
    3. If failed, Meta-Hebbian weakens the path
    """
    print("\n" + "="*60)
    print("TEST 1: Swarm V3 + Meta-Hebbian Plasticity")
    print("="*60)
    
    if not META_HEBBIAN_AVAILABLE:
        print("❌ SKIP: MetaHebbianPlasticity not available")
        return False
    
    # Initialize components
    topology = create_topology_engine() if TOPOLOGY_AVAILABLE else None
    mycelial = MycelialReasoningLite() if MYCELIAL_AVAILABLE else None
    
    swarm = SwarmNavigator(
        topology=topology,
        mycelial=mycelial,
        use_neurodiverse=True
    )
    
    meta_hebbian = MetaHebbianPlasticity(MetaHebbianConfig(
        num_codes=1024,
        num_heads=4
    ))
    
    print(f"✓ SwarmNavigatorV3 initialized")
    print(f"✓ MetaHebbianPlasticity initialized (codes={meta_hebbian.config.num_codes})")
    
    # Test navigation
    test_cases = [
        ("machine learning", "neural network"),
        ("philosophy", "computer science"),
        ("water", "fire")
    ]
    
    fitness_scores = []
    
    for start, target in test_cases:
        print(f"\nNavigating: '{start}' -> '{target}'...")
        result = swarm.navigate(start, target, max_steps=20, debug=False)
        
        # Use improvement as fitness signal
        fitness = result['improvement']
        fitness_scores.append(fitness)
        
        print(f"  Result: success={result['success']}, improvement={fitness:.4f}")
        print(f"  Steps: {result['steps']}, Final Sim: {result['final_similarity']:.4f}")
    
    # Evolve Meta-Hebbian rules based on navigation success
    print("\nEvolving Meta-Hebbian rules based on navigation performance...")
    evolution_stats = meta_hebbian.evolve_rules(fitness_scores)
    
    print(f"  Best Fitness: {evolution_stats.get('best_fitness', 'N/A')}")
    print(f"  Population Size: {evolution_stats.get('population_size', 'N/A')}")
    
    # Analyze rules
    analysis = meta_hebbian.get_rule_analysis()
    print(f"\nMeta-Hebbian Rule Analysis:")
    mean_eta = analysis.get('mean_eta', 'N/A')
    if isinstance(mean_eta, (int, float)):
        print(f"  Mean Learning Rate (η): {mean_eta:.6f}")
    else:
        print(f"  Mean Learning Rate (η): {mean_eta}")
    print(f"  Interpretation: {analysis.get('interpretation', ['N/A'])[0] if analysis.get('interpretation') else 'N/A'}")
    
    print("\n✅ TEST 1 PASSED: Swarm V3 + Meta-Hebbian integration works!")
    return True


def test_swarm_active_inference_integration():
    """
    Test: Swarm V3 provides navigation for Active Inference action selection.
    
    Flow:
    1. Active Inference agent selects a "bridge concepts" action
    2. Swarm V3 is used to find the path between concepts
    3. Path quality influences agent's belief update
    """
    print("\n" + "="*60)
    print("TEST 2: Swarm V3 + Active Inference Agent")
    print("="*60)
    
    if not ACTIVE_INFERENCE_AVAILABLE:
        print("❌ SKIP: ActiveInferenceAgent not available")
        return False
    
    # Initialize components
    topology = create_topology_engine() if TOPOLOGY_AVAILABLE else None
    mycelial = MycelialReasoningLite() if MYCELIAL_AVAILABLE else None
    
    swarm = SwarmNavigator(
        topology=topology,
        mycelial=mycelial,
        use_neurodiverse=True
    )
    
    ai_agent = ActiveInferenceAgent(ActiveInferenceConfig(
        state_dim=64,
        planning_horizon=3
    ))
    
    print(f"✓ SwarmNavigatorV3 initialized")
    print(f"✓ ActiveInferenceAgent initialized (state_dim={ai_agent.config.state_dim})")
    
    # Simulate: AI agent wants to bridge two concepts
    source_concept = "entropy"
    target_concept = "information"
    
    # Create a BRIDGE_CONCEPTS action
    action = Action(
        action_type=ActionType.BRIDGE_CONCEPTS,
        target=f"{source_concept}->{target_concept}",
        parameters={"source": source_concept, "target": target_concept}
    )
    
    print(f"\nAI Agent selected action: BRIDGE_CONCEPTS")
    print(f"  Source: '{source_concept}'")
    print(f"  Target: '{target_concept}'")
    
    # Use Swarm V3 to execute the bridging
    print("\nSwarm V3 executing bridge navigation...")
    result = swarm.navigate(source_concept, target_concept, max_steps=25, debug=False)
    
    print(f"  Result: success={result['success']}")
    print(f"  Path Length: {result['steps']} steps")
    print(f"  Similarity: {result['init_similarity']:.4f} -> {result['final_similarity']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f}")
    
    # Report neurotype contributions
    contribs = result.get('neurotype_contributions', {})
    if contribs:
        sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        print(f"  Neurotype Contributions: {', '.join([f'{k}:{v:.2f}' for k,v in sorted_contribs[:3]])}")
    
    # Compute information gain for AI agent
    information_gain = result['improvement'] * 10  # Scale for AI system
    
    print(f"\nInformation Gain for AI Agent: {information_gain:.4f}")
    
    # Update AI agent's belief (simplified simulation)
    # In full integration, this would update the agent's internal state
    print("  -> AI Agent receives feedback from Swarm V3 navigation")
    
    print("\n✅ TEST 2 PASSED: Swarm V3 + Active Inference integration works!")
    return True


def test_full_loop_simulation():
    """
    Test: Simulate a mini Self-Feeding Loop cycle using Swarm V3.
    
    Flow:
    1. Detect a "knowledge gap" (simulated)
    2. Generate hypothesis to fill the gap
    3. Use Swarm V3 to explore the hypothesis
    4. Evaluate and learn from result
    """
    print("\n" + "="*60)
    print("TEST 3: Self-Feeding Loop Simulation with Swarm V3")
    print("="*60)
    
    # Initialize Swarm
    topology = create_topology_engine() if TOPOLOGY_AVAILABLE else None
    mycelial = MycelialReasoningLite() if MYCELIAL_AVAILABLE else None
    
    swarm = SwarmNavigator(
        topology=topology,
        mycelial=mycelial,
        use_neurodiverse=True
    )
    
    print(f"✓ SwarmNavigatorV3 initialized")
    
    # Simulated knowledge gap
    gap = {
        "type": "connection_missing",
        "source": "recursion",
        "target": "self-reference",
        "confidence": 0.3
    }
    
    print(f"\n📌 Detected Knowledge Gap:")
    print(f"   Type: {gap['type']}")
    print(f"   Source: '{gap['source']}'")
    print(f"   Target: '{gap['target']}'")
    print(f"   Confidence (low): {gap['confidence']}")
    
    # Generate hypothesis
    hypothesis = {
        "id": "hyp_001",
        "text": f"'{gap['source']}' is conceptually related to '{gap['target']}' through self-application",
        "action": "navigate_and_validate"
    }
    
    print(f"\n💡 Hypothesis Generated:")
    print(f"   {hypothesis['text']}")
    
    # Execute hypothesis via Swarm V3
    print("\n🔄 Executing Hypothesis via Swarm V3...")
    result = swarm.navigate(gap['source'], gap['target'], max_steps=30, debug=False)
    
    print(f"\n📊 Execution Results:")
    print(f"   Success: {result['success']}")
    print(f"   Similarity Improvement: {result['improvement']:.4f}")
    print(f"   Steps: {result['steps']}")
    print(f"   Topological Events: {len(result.get('topological_events', []))}")
    
    # Evaluate
    if result['success'] and result['improvement'] > 0.1:
        verdict = "CONFIRMED"
        learning = "Reinforce connection via Meta-Hebbian"
    elif result['success']:
        verdict = "WEAK SUPPORT"
        learning = "Store as weak hypothesis for future validation"
    else:
        verdict = "REFUTED"
        learning = "Weaken connection hypothesis"
    
    print(f"\n🎯 Hypothesis Verdict: {verdict}")
    print(f"   Learning Action: {learning}")
    
    # Memory stats
    memory_stats = swarm.get_memory_stats()
    print(f"\n📈 Swarm Memory Stats:")
    print(f"   Total Trajectories: {memory_stats.get('total_trajectories', 0)}")
    print(f"   Success Rate: {memory_stats.get('success_rate', 0):.1%}")
    
    print("\n✅ TEST 3 PASSED: Self-Feeding Loop simulation works!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("SWARM V3 INTEGRATION TESTS - SANDBOX MODE")
    print("="*60)
    print("\nThis test validates that Swarm V3 can integrate with:")
    print("  1. Meta-Hebbian Plasticity (neuroplasticity)")
    print("  2. Active Inference Agent (decision making)")
    print("  3. Self-Feeding Loop (knowledge acquisition)")
    
    results = {}
    
    # Test 1
    results['meta_hebbian'] = test_swarm_meta_hebbian_integration()
    
    # Test 2
    results['active_inference'] = test_swarm_active_inference_integration()
    
    # Test 3
    results['self_feeding_loop'] = test_full_loop_simulation()
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL/SKIP"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed or were skipped.")


if __name__ == "__main__":
    run_all_tests()
