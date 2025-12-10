# tests/test_system_integration.py
"""
Alexandria :: System Integration Tests

Tests the integration between PreStructuralField and MycelialReasoning,
specifically the feedback loop where geometry reinforces the network.

Usage:
    python -m pytest tests/test_system_integration.py -v
"""

import numpy as np
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.field.pre_structural_field import PreStructuralField
from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig


def _make_dummy_embedding(dim: int = 384) -> np.ndarray:
    """Creates a normalized random embedding."""
    vec = np.random.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def test_field_crystallization_updates_mycelial():
    """
    Integration test: Field -> Mycelial feedback loop.
    
    Steps:
    1. Create PreStructuralField + MycelialReasoning
    2. Generate two points in the manifold with distinct VQ-VAE codes
    3. Build a simple graph (2 nodes, 1 edge)
    4. Call _incorporate_to_mycelial
    5. Verify that the Mycelial graph gained new/stronger connections
    """

    # ------------------------------------------------------------------
    # 1) Initialize real components
    # ------------------------------------------------------------------
    mycelial_config = MycelialConfig(
        save_path="/tmp/test_mycelial_integration.pkl"
    )
    mycelial = MycelialReasoning(mycelial_config)
    # Start fresh
    mycelial.reset()

    field = PreStructuralField()
    field.connect_mycelial(mycelial)

    base_dim = field.config.base_dim

    # ------------------------------------------------------------------
    # 2) Create two distinct points with different codes
    # ------------------------------------------------------------------
    codes_a = np.array([10, 20, 30, 40], dtype=np.int32)
    codes_b = np.array([11, 21, 31, 41], dtype=np.int32)

    emb_a = _make_dummy_embedding(base_dim)
    emb_b = _make_dummy_embedding(base_dim)

    # Trigger points into the manifold
    field.trigger(embedding=emb_a, codes=codes_a, intensity=1.0)
    field.trigger(embedding=emb_b, codes=codes_b, intensity=1.0)

    # Snapshot before crystallization
    stats_before = mycelial.get_network_stats()
    edges_before = stats_before.get("active_edges", 0)

    # ------------------------------------------------------------------
    # 3) Build a synthetic graph using real manifold point coordinates
    # ------------------------------------------------------------------
    active_points = list(field.manifold.points.items())
    assert len(active_points) >= 2, "Expected at least two points in manifold"

    (id_a, p_a), (id_b, p_b) = active_points[0], active_points[1]

    graph = {
        "nodes": [
            {
                "id": "na",
                "coordinates": p_a.coordinates.tolist(),
                "free_energy": float(field.get_free_energy_at(p_a.coordinates)),
            },
            {
                "id": "nb",
                "coordinates": p_b.coordinates.tolist(),
                "free_energy": float(field.get_free_energy_at(p_b.coordinates)),
            },
        ],
        "edges": [
            {
                "source": "na",
                "target": "nb",
                "weight": 1.0,
            }
        ],
    }

    # ------------------------------------------------------------------
    # 4) Incorporate into Mycelial via PreStructuralField
    # ------------------------------------------------------------------
    field._incorporate_to_mycelial(graph)

    # ------------------------------------------------------------------
    # 5) Verify that the Mycelial graph was updated
    # ------------------------------------------------------------------
    stats_after = mycelial.get_network_stats()
    edges_after = stats_after.get("active_edges", 0)

    assert edges_after >= edges_before, (
        f"Expected edges to not decrease ({edges_before} -> {edges_after})"
    )
    
    # Additionally, verify that at least some edge was added
    if edges_before == 0:
        assert edges_after > 0, "Expected new edges to be created"


def test_field_crystallize_integrates_with_mycelial():
    """
    Integration test: Full crystallize() flow.
    
    Uses the real crystallize() method which internally calls
    _incorporate_to_mycelial.
    """
    
    # Initialize components
    mycelial_config = MycelialConfig(
        save_path="/tmp/test_mycelial_crystallize.pkl"
    )
    mycelial = MycelialReasoning(mycelial_config)
    mycelial.reset()

    field = PreStructuralField()
    field.connect_mycelial(mycelial)

    base_dim = field.config.base_dim

    # Add multiple points to create meaningful attractors
    for i in range(5):
        codes = np.array([10 + i, 20 + i, 30 + i, 40 + i], dtype=np.int32)
        emb = _make_dummy_embedding(base_dim)
        field.trigger(embedding=emb, codes=codes, intensity=1.0)

    stats_before = mycelial.get_network_stats()

    # Run crystallize (this calls _incorporate_to_mycelial internally)
    graph = field.crystallize()

    stats_after = mycelial.get_network_stats()

    # Basic assertions
    assert isinstance(graph, dict)
    assert "nodes" in graph
    assert "edges" in graph
    
    # If there were edges, verify propagation to Mycelial
    if len(graph.get("edges", [])) > 0:
        assert stats_after.get("active_edges", 0) >= stats_before.get("active_edges", 0)


if __name__ == "__main__":
    print("Running: test_field_crystallization_updates_mycelial")
    test_field_crystallization_updates_mycelial()
    print("✅ PASSED")

    print("Running: test_field_crystallize_integrates_with_mycelial")
    test_field_crystallize_integrates_with_mycelial()
    print("✅ PASSED")

    print("\n✅ All integration tests passed!")
