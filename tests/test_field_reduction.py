import sys
import numpy as np
import pytest
sys.path.insert(0, '.')

from core.field import PreStructuralField, PreStructuralConfig

def test_dim_reduction_integration():
    """Testa se o field aceita input 384d e reduz para 32d automaticamente."""
    
    # Configuração 32d
    config = PreStructuralConfig(
        base_dim=32,
        input_dim=384
    )
    field = PreStructuralField(config)
    
    # Input 384d
    embedding_384 = np.random.randn(384)
    state = field.trigger(embedding_384)
    
    # Verifica se o ponto no manifold é 32d
    assert field.manifold.current_dim == 32
    for point in field.manifold.points.values():
        assert point.coordinates.shape == (32,)
        break
    
    print("Test passed: 384d -> 32d reduction working")

if __name__ == "__main__":
    test_dim_reduction_integration()
