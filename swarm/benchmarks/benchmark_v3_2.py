"""
swarm/benchmarks/benchmark_v3_2.py

Benchmark for Swarm v3.2 features:
1. SwarmAction parsing and execution
2. ActiveBridgeBuilder generation
3. AdaptiveEarlyStoppingCommittee mode-based behavior
"""

import sys
import os
# Add project root to path for module resolution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Test imports
print("=" * 60)
print("SWARM V3.2 BENCHMARK")
print("=" * 60)

# =============================================================================
# TEST 1: SwarmAction & ActionType
# =============================================================================
print("\n" + "=" * 60)
print("TEST 1: SwarmAction & ActionType")
print("=" * 60)

try:
    from swarm.core import SwarmAction, ActionType
    
    # Create a bridge action
    action = SwarmAction(
        type=ActionType.BRIDGE_CONCEPTS,
        start=np.random.randn(384),
        target=np.random.randn(384),
        params={'mode': 'creative'},
        constraints={'max_steps': 50}
    )
    
    print(f"✓ Created SwarmAction: type={action.type.value}, id={action.id}")
    print(f"  Params: {action.params}")
    print(f"  Constraints: {action.constraints}")
    print("\n✅ SwarmAction TEST PASSED")
    test1_pass = True
except Exception as e:
    print(f"❌ SwarmAction TEST FAILED: {e}")
    test1_pass = False

# =============================================================================
# TEST 2: ActiveBridgeBuilder
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: ActiveBridgeBuilder")
print("=" * 60)

try:
    from swarm.topology.bridge import ActiveBridgeBuilder, BridgeCandidate
    
    # Initialize without topology engine (uses fallback)
    builder = ActiveBridgeBuilder()
    
    # Create two distant concepts
    concept_a = np.random.randn(384)
    concept_a = concept_a / np.linalg.norm(concept_a)
    
    concept_b = np.random.randn(384)
    concept_b = concept_b / np.linalg.norm(concept_b)
    
    # Propose bridges via interpolation
    candidates = builder.propose_bridge(concept_a, concept_b, method='interpolation')
    
    print(f"✓ Generated {len(candidates)} bridge candidates")
    for i, c in enumerate(candidates):
        print(f"  {i+1}. Method: {c.method}, Confidence: {c.confidence:.3f}")
    
    # Validate a candidate
    if candidates:
        is_valid, reason = builder.validate_bridge(candidates[0])
        print(f"✓ Validation result: valid={is_valid}, reason={reason}")
    
    print("\n✅ ActiveBridgeBuilder TEST PASSED")
    test2_pass = True
except Exception as e:
    print(f"❌ ActiveBridgeBuilder TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    test2_pass = False

# =============================================================================
# TEST 3: AdaptiveEarlyStoppingCommittee
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: AdaptiveEarlyStoppingCommittee")
print("=" * 60)

try:
    from swarm.tools.early_stopping import AdaptiveEarlyStoppingCommittee, PathQualityVoter
    
    committee = AdaptiveEarlyStoppingCommittee()
    
    # Create a trajectory that converges
    target = np.random.randn(384)
    target = target / np.linalg.norm(target)
    
    trajectory = []
    start = np.random.randn(384)
    start = start / np.linalg.norm(start)
    
    # Simulate gradual convergence
    for i in range(20):
        t = i / 19
        pos = (1 - t) * start + t * target
        pos = pos / np.linalg.norm(pos)
        trajectory.append(pos)
    
    # Test with different modes
    for mode in ['sprint', 'balanced', 'creative']:
        decision = committee.should_stop(
            trajectory=trajectory,
            target=target,
            current_step=len(trajectory),
            context={'mode': mode}
        )
        print(f"✓ Mode={mode:8s} -> stop={decision.should_stop}, agreement={decision.agreement_ratio:.2f}")
        for v in decision.votes:
            print(f"    {v.voter_name}: stop={v.should_stop}, confidence={v.confidence:.2f}")
    
    print("\n✅ AdaptiveEarlyStoppingCommittee TEST PASSED")
    test3_pass = True
except Exception as e:
    print(f"❌ AdaptiveEarlyStoppingCommittee TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    test3_pass = False

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

results = {
    'swarm_action': test1_pass,
    'bridge_builder': test2_pass,
    'adaptive_stopping': test3_pass
}

for name, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {name}: {status}")

total = sum(results.values())
print(f"\nTotal: {total}/{len(results)} tests passed")

if total == len(results):
    print("\n🎉 All v3.2 features verified!")
    sys.exit(0)
else:
    sys.exit(1)
