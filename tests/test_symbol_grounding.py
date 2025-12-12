
"""
Test Symbol Grounding
=====================

Verifies:
1. Grounding text returns valid (head, code) tuples.
2. Ground Integration with Mycelial.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.reasoning.symbol_grounding import SymbolGrounder
from core.reasoning.mycelial_reasoning import MycelialReasoning

class MockTopology:
    def encode(self, texts):
        # Return fake vector of size 64
        return [[0.1] * 64 for _ in texts]

class MockVQVAE:
    def __init__(self):
        self.vqvae = MagicMock()
        self.vqvae.device = 'cpu'
        
    def encode(self, tensor):
        # Return fake indices
        import torch
        # 4 heads, 1 code each
        return torch.tensor([[10, 20, 30, 40]])

def test_grounding_mocked():
    """Test grounding logic with mocks"""
    grounder = SymbolGrounder(
        topology_engine=MockTopology(),
        vqvae_wrapper=MockVQVAE()
    )
    
    nodes = grounder.ground("test concept")
    
    assert len(nodes) == 4
    assert nodes[0] == (0, 10)
    assert nodes[1] == (1, 20)

def test_grounding_integration_real_imports():
    """Test that we can init with real imports (if available) or handle missing"""
    # This just tests the imports don't crash the constructor
    grounder = SymbolGrounder()
    assert grounder is not None

def test_executor_grounding_flow():
    """Test that HypothesisExecutor uses grounded nodes"""
    from core.loop.hypothesis_executor import HypothesisExecutor, ExecutableAction, ExecutionActionType
    
    executor = HypothesisExecutor(mycelial_reasoning=MagicMock())
    executor.grounder = MagicMock()
    executor.grounder.ground.return_value = [(0, 10), (0, 11)] # Two nodes in same head
    
    # Action with text but no nodes
    action = ExecutableAction(
        action_type=ExecutionActionType.BRIDGE_CONCEPTS,
        target="A <-> B",
        parameters={"source": "A", "target": "B"}
    )
    
    # Mock mycelial connection return
    executor.mycelial.connect_nodes.return_value = 0.5
    
    result = executor._execute_action(action)
    
    # Should have called ground("A") and ground("B")
    assert executor.grounder.ground.call_count >= 2
    # Should have called connect_nodes
    executor.mycelial.connect_nodes.assert_called()
    assert "Grounded Bridge" in result.evidence_found[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
