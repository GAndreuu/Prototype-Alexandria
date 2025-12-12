
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("1. Testing Meta-Hebbian Import...")
    from core.learning.meta_hebbian import create_meta_hebbian_system
    meta = create_meta_hebbian_system(num_codes=384, num_heads=4, load_existing=False)
    print("   ✅ Meta-Hebbian instantiated.")

    print("\n2. Testing Predictive Coding Import...")
    from core.learning.predictive_coding import create_predictive_coding_system
    pc = create_predictive_coding_system(input_dim=384, code_dim=384, load_existing=False)
    print("   ✅ Predictive Coding instantiated (384D).")

    print("\n3. Testing Active Inference Import...")
    from core.learning.active_inference import create_active_inference_system
    # Active Inference uses PC internally
    ai_system = create_active_inference_system(state_dim=384, load_existing=False, use_predictive_coding=True)
    print("   ✅ Active Inference instantiated.")

    print("\n4. Testing Synergy...")
    # Simulate a cycle
    obs = np.random.randn(384)
    # Normalize
    obs = obs / np.linalg.norm(obs)
    
    result = ai_system.perception_action_cycle(external_observation=obs)
    print(f"   ✅ Cycle 1 completed. Action: {result['action_taken']['type']}")
    
    print("\n🎉 ALL SYSTEMS GO.")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
