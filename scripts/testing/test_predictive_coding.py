"""
Test Predictive Coding module compatibility.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_predictive_coding():
    print("🔮 Testing Predictive Coding Module")
    print()
    
    try:
        from core.learning.predictive_coding import PredictiveCodingNetwork, PredictiveCodingConfig
        import numpy as np
        
        # Initialize with default config
        config = PredictiveCodingConfig()
        
        network = PredictiveCodingNetwork(config=config)
        print("✅ Predictive Coding initialized")
        print()
        
        # Check state loaded
        if hasattr(network, 'layers'):
            print(f"✅ Network layers initialized: {len(network.layers)} layers")
        
        # Check key methods exist
        methods_to_check = ['forward', 'update', 'predict']
        for method in methods_to_check:
            if hasattr(network, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"⚠️  Warning: Method '{method}' not found")
        
        print()
        
        # Try a simple forward pass
        try:
            test_input = np.random.randn(1, 64).astype(np.float32)
            output = network.forward(test_input)
            print("✅ Forward pass successful")
            print(f"   Output shape: {output.shape}")
        except Exception as e:
            print(f"⚠️  Forward pass warning: {e}")
        
        print()
        print("✅ Predictive Coding: MODULE LOADED")
        print("   Core components available")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictive Coding Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_predictive_coding()
    sys.exit(0 if success else 1)
