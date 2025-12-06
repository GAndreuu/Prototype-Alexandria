import sys
sys.path.append('c:\\Users\\G\\Desktop\\Alexandria')
sys.path.append('c:\\Users\\G\\Desktop\\Alexandria\\core\\learning')
from core.learning.integration_layer import IntegrationConfig, AlexandriaIntegratedPipeline, MultiAgentOrchestrator, SystemProfile
import time

def verify_nemesis_stack():
    print("="*60)
    print("⚔️  VERIFYING COGNITIVE NEMESIS STACK")
    print("="*60)
    
    # 1. Initialize Pipeline (LITE MODE)
    print("\n[1] Initializing Pipeline (LITE Profile)...")
    config = IntegrationConfig(profile=SystemProfile.LITE)
    pipeline = AlexandriaIntegratedPipeline(config)
    
    # Verify Limits
    assert config.resources.max_memory_mb == 8192
    assert config.pc_num_iterations == 3
    print("   ✅ Lite Mode Configured")
    
    # 2. Initialize Orchestrator
    print("\n[2] Spawning Multi-Agent Orchestrator...")
    orchestrator = MultiAgentOrchestrator(pipeline)
    
    # Verify Agents
    expected_agents = {'scout', 'judge', 'weaver'}
    assert set(orchestrator.agents.keys()) == expected_agents
    print(f"   ✅ Agents Spawned: {list(orchestrator.agents.keys())}")
    
    # 3. Simulate Cycle
    print("\n[3] Running Cognitive Cycle (Input: 'The concept of entropy in information theory')...")
    start = time.time()
    results = orchestrator.run_cycle("The concept of entropy in information theory")
    duration = time.time() - start
    
    print(f"   ✅ Cycle Completed in {duration:.2f}s")
    print(f"   ✅ Perception: {results.get('perception', {}).get('text_length')} chars")
    
    for agent in expected_agents:
        actions = results.get(agent, [])
        if actions:
            print(f"   🤖 {agent.upper()}: Executed {len(actions)} actions")
            print(f"      Action: {actions[0]['action']}")
    
    print("\n✅ VERIFICATION COMPLETE: NEMESIS CORE OPERATIONAL")

if __name__ == "__main__":
    verify_nemesis_stack()
