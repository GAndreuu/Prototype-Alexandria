"""
Real Data Integration Test for Swarm v3.2

Tests the SwarmNavigator with real TopologyEngine and embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

print("=" * 60)
print("SWARM V3.2 - REAL DATA INTEGRATION TEST")
print("=" * 60)

# =============================================================================
# 1. Load TopologyEngine
# =============================================================================
print("\n[1] Loading TopologyEngine...")

try:
    from core.topology.topology_engine import TopologyEngine, create_topology_engine
    
    topology = create_topology_engine()
    print(f"✓ TopologyEngine loaded: model={topology.model_name if hasattr(topology, 'model_name') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to load TopologyEngine: {e}")
    topology = None

# =============================================================================
# 2. Generate real embeddings for test concepts
# =============================================================================
print("\n[2] Generating real embeddings...")

test_concepts = [
    "Neural networks are computational models inspired by biological brains",
    "Machine learning algorithms learn patterns from data",
    "Deep learning uses multiple layers of neural networks",
    "Quantum computing uses quantum mechanical phenomena",
    "Classical computers use binary logic gates"
]

try:
    if topology:
        embeddings = topology.encode(test_concepts)
        print(f"✓ Generated {len(embeddings)} embeddings, shape: {embeddings.shape}")
        
        # Show pairwise similarities
        print("\nPairwise similarities:")
        for i in range(len(test_concepts)):
            for j in range(i+1, len(test_concepts)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  [{i}]-[{j}]: {sim:.3f}")
    else:
        # Fallback: random embeddings
        embeddings = np.random.randn(len(test_concepts), 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        print("✓ Using fallback random embeddings")
except Exception as e:
    print(f"✗ Embedding generation failed: {e}")
    import traceback
    traceback.print_exc()
    embeddings = None

# =============================================================================
# 3. Test SwarmAction with real embeddings
# =============================================================================
print("\n[3] Testing SwarmAction with real embeddings...")

if embeddings is not None:
    try:
        from swarm.core import SwarmAction, ActionType
        
        # Bridge neural networks to quantum computing
        start_emb = embeddings[0]  # Neural networks
        target_emb = embeddings[3]  # Quantum computing
        
        action = SwarmAction(
            type=ActionType.BRIDGE_CONCEPTS,
            start=start_emb,
            target=target_emb,
            params={'mode': 'creative'},
            constraints={'max_steps': 30}
        )
        
        initial_sim = np.dot(start_emb, target_emb) / (
            np.linalg.norm(start_emb) * np.linalg.norm(target_emb)
        )
        print(f"✓ Created BRIDGE_CONCEPTS action")
        print(f"  Start: '{test_concepts[0][:40]}...'")
        print(f"  Target: '{test_concepts[3][:40]}...'")
        print(f"  Initial similarity: {initial_sim:.3f}")
        
    except Exception as e:
        print(f"✗ SwarmAction creation failed: {e}")
        action = None
else:
    action = None

# =============================================================================
# 4. Test ActiveBridgeBuilder with real embeddings
# =============================================================================
print("\n[4] Testing ActiveBridgeBuilder with real embeddings...")

if embeddings is not None:
    try:
        from swarm.topology.bridge import ActiveBridgeBuilder
        
        builder = ActiveBridgeBuilder(topology_engine=topology)
        
        # Find bridges between neural networks and quantum computing
        candidates = builder.propose_bridge(
            embeddings[0],  # Neural networks
            embeddings[3],  # Quantum computing
            method='interpolation',
            num_candidates=3
        )
        
        print(f"✓ Generated {len(candidates)} bridge candidates")
        for i, c in enumerate(candidates):
            print(f"  {i+1}. Confidence: {c.confidence:.3f}, Method: {c.method}")
            
            # Validate
            is_valid, reason = builder.validate_bridge(c)
            print(f"     Valid: {is_valid}, Reason: {reason}")
            
    except Exception as e:
        print(f"✗ ActiveBridgeBuilder test failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 5. Test Full Navigation with SwarmNavigator
# =============================================================================
print("\n[5] Testing SwarmNavigator.navigate_action()...")

if embeddings is not None and action is not None:
    try:
        from swarm.navigator import SwarmNavigator
        
        # Create navigator with topology
        navigator = SwarmNavigator(
            topology_engine=topology,
            use_neurodiverse=True
        )
        
        # Execute the action
        result = navigator.navigate_action(action, debug=True)
        
        print(f"✓ Navigation completed!")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Initial similarity: {result['init_similarity']:.3f}")
        print(f"  Final similarity: {result['final_similarity']:.3f}")
        print(f"  Improvement: {result['improvement']:.3f}")
        print(f"  Mode used: {result['mode']}")
        
        if result.get('neurotype_contributions'):
            print(f"  Neurotype contributions:")
            for nt, contrib in result['neurotype_contributions'].items():
                print(f"    {nt}: {contrib:.2f}")
                
    except Exception as e:
        print(f"✗ SwarmNavigator test failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
