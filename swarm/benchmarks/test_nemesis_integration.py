"""
Test: Swarm ↔ Nemesis Integration

Demonstrates the full pipeline:
1. User provides concepts
2. Swarm navigates
3. Nemesis generates explanation
4. User can provide feedback
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("SWARM ↔ NEMESIS INTEGRATION TEST")
print("=" * 60)

# 1. Setup
print("\n[1] Setting up integration...")
from core.topology.topology_engine import create_topology_engine
from swarm.integrations.nemesis import create_swarm_nemesis_integration

topology = create_topology_engine()
integration = create_swarm_nemesis_integration(topology_engine=topology)
print("✓ Integration created")

# 2. Navigate with explanation
print("\n[2] Navigating 'Neural Networks' → 'Quantum Computing'...")
print("-" * 60)

result = integration.navigate_and_explain(
    start_concept="Neural networks are computational models inspired by biological brains",
    target_concept="Quantum computing uses quantum mechanical phenomena for computation",
    generate_explanation=True
)

# 3. Display explanation
print("\n" + "=" * 60)
print("EXPLANATION FOR USER")
print("=" * 60)

explanation = result.explanation
print(f"\n{explanation.summary}")
print(f"\n📊 {explanation.path_description}")
print(f"\n🧠 {explanation.neurotype_narrative}")
print(f"\n💡 Insights:")
for insight in explanation.actionable_insights:
    print(f"   • {insight}")

print(f"\n📈 Confiança: {explanation.confidence:.0%}")

# 4. Show raw data
print("\n" + "-" * 60)
print("RAW NAVIGATION DATA:")
print("-" * 60)
sr = result.swarm_result
print(f"  Success: {sr['success']}")
print(f"  Steps: {sr['steps']}")
print(f"  Similarity: {sr['init_similarity']:.3f} → {sr['final_similarity']:.3f}")
print(f"  Improvement: {sr['improvement']:.3f}")

# 5. Simulate feedback
print("\n" + "=" * 60)
print("FEEDBACK LOOP")
print("=" * 60)
print("\n[Simulating positive user feedback...]")
integration.receive_feedback(
    navigation_id=0,
    positive=True,
    comment="A conexão faz sentido!"
)
print("✓ Feedback recorded")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
