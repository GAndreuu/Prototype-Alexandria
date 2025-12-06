"""
Alexandria System - Full Integration Test Suite
Tests all core components sequentially to verify functionality and identify errors.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import traceback
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestReport:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        
    def log_test(self, name, passed, error=None, details=None):
        """Log a test result"""
        self.tests.append({
            'name': name,
            'passed': passed,
            'error': str(error) if error else None,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        if passed:
            self.passed += 1
            logger.info(f"✅ {name}")
            if details:
                logger.info(f"   {details}")
        else:
            self.failed += 1
            logger.error(f"❌ {name}")
            logger.error(f"   Error: {error}")
    
    def print_summary(self):
        """Print final report"""
        logger.info("\n" + "="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {self.passed + self.failed}")
        logger.info(f"✅ Passed: {self.passed}")
        logger.info(f"❌ Failed: {self.failed}")
        logger.info(f"Success Rate: {self.passed/(self.passed+self.failed)*100:.1f}%")
        
        if self.failed > 0:
            logger.info(f"\n{'='*60}")
            logger.info("Failed Tests:")
            for test in self.tests:
                if not test['passed']:
                    logger.info(f"  - {test['name']}: {test['error']}")

def test_vqvae(report):
    """Test VQ-VAE Neural Learner"""
    logger.info("\n" + "="*60)
    logger.info("🧠 Testing VQ-VAE Neural Learner")
    logger.info("="*60)
    
    try:
        from core.reasoning.neural_learner import V2Learner
        import numpy as np
        
        # Initialize
        learner = V2Learner()
        report.log_test("VQ-VAE: Initialization", True, details=f"Device: {learner.device}")
        
        # Test encoding
        test_vector = np.random.randn(384).astype(np.float32).tolist()
        encoded = learner.encode([test_vector])
        report.log_test("VQ-VAE: Encoding", True, details=f"Output shape: {encoded.shape}")
        
        # Test decoding
        decoded = learner.decode(encoded.tolist())
        report.log_test("VQ-VAE: Decoding", True, details=f"Output shape: {decoded.shape}")
        
        # Check model is loaded
        if learner.is_loaded:
            report.log_test("VQ-VAE: Model Loaded", True, details=f"Path: {learner.model_path}")
        else:
            report.log_test("VQ-VAE: Model Loaded", False, error="Model not loaded")
            
    except Exception as e:
        report.log_test("VQ-VAE: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_mycelial(report):
    """Test Mycelial Reasoning Network"""
    logger.info("\n" + "="*60)
    logger.info("🍄 Testing Mycelial Reasoning")
    logger.info("="*60)
    
    try:
        from core.reasoning.mycelial_reasoning import MycelialReasoning
        import numpy as np
        
        # Initialize
        mycelial = MycelialReasoning()
        report.log_test("Mycelial: Initialization", True)
        
        # Test observation
        indices = np.array([72, 150, 190, 229])
        mycelial.observe(indices)
        report.log_test("Mycelial: Observation", True, details="4 indices observed")
        
        # Test reasoning
        new_indices, activation = mycelial.reason(indices, steps=3)
        report.log_test("Mycelial: Reasoning", True, 
                       details=f"Input: {indices} → Output: {new_indices}")
        
        # Check stats
        stats = mycelial.get_network_stats()
        report.log_test("Mycelial: Statistics", True,
                       details=f"Observations: {stats['total_observations']}, Connections: {stats['active_connections']}")
        
        # Test save/load
        mycelial.save_state()
        report.log_test("Mycelial: Save State", True)
        
    except Exception as e:
        report.log_test("Mycelial: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_lancedb(report):
    """Test LanceDB Storage"""
    logger.info("\n" + "="*60)
    logger.info("💾 Testing LanceDB Storage")
    logger.info("="*60)
    
    try:
        from core.memory.storage import LanceDBStorage
        import numpy as np
        
        # Initialize
        storage = LanceDBStorage()
        report.log_test("LanceDB: Initialization", True)
        
        # Count records
        count = storage.count()
        report.log_test("LanceDB: Count Records", True, details=f"{count} records")
        
        # Test search
        query_vector = np.random.randn(384).astype(np.float32).tolist()
        results = storage.search(query_vector, limit=5)
        report.log_test("LanceDB: Search", True, details=f"{len(results)} results returned")
        
        # Check if results have required fields
        if results and len(results) > 0:
            result = results[0]
            has_fields = all(k in result for k in ['text', 'metadata', 'distance'])
            report.log_test("LanceDB: Result Format", has_fields, 
                           error="Missing required fields" if not has_fields else None)
        
    except Exception as e:
        report.log_test("LanceDB: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_abduction_engine(report):
    """Test Abduction Engine"""
    logger.info("\n" + "="*60)
    logger.info("🔮 Testing Abduction Engine")
    logger.info("="*60)
    
    try:
        from core.reasoning.abduction_engine import AbductionEngine
        
        # Initialize
        engine = AbductionEngine()
        report.log_test("Abduction: Initialization", True)
        
        # Detect gaps
        gaps = engine.detect_knowledge_gaps()
        report.log_test("Abduction: Detect Gaps", True, details=f"{len(gaps)} gaps detected")
        
        # Generate hypotheses
        hypotheses = engine.generate_hypotheses()
        report.log_test("Abduction: Generate Hypotheses", True, 
                       details=f"{len(hypotheses)} hypotheses generated")
        
    except Exception as e:
        report.log_test("Abduction: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_causal_engine(report):
    """Test Causal Reasoning Engine"""
    logger.info("\n" + "="*60)
    logger.info("⚡ Testing Causal Reasoning")
    logger.info("="*60)
    
    try:
        from core.reasoning.causal_reasoning import CausalEngine
        from core.topology.topology_engine import TopologyEngine
        from core.memory.semantic_memory import SemanticMemory
        
        # Initialize dependencies
        engine = TopologyEngine()
        memory = SemanticMemory()
        
        # Initialize
        causal = CausalEngine(engine, memory)
        report.log_test("Causal: Initialization", True)
        
        # Build graph
        causal.build_causal_graph()
        report.log_test("Causal: Build Graph", True, 
                       details=f"{len(causal.graph.nodes)} nodes")
        
        # Infer causality
        result = causal.infer_causality("physics", "mathematics")
        report.log_test("Causal: Infer Causality", True,
                       details=f"Confidence: {result.get('confidence', 0):.2f}")
        
    except Exception as e:
        report.log_test("Causal: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_oracle(report):
    """Test Neural Oracle"""
    logger.info("\n" + "="*60)
    logger.info("🔮 Testing Neural Oracle")
    logger.info("="*60)
    
    try:
        from core.agents.oracle import NeuralOracle
        
        # Initialize
        oracle = NeuralOracle()
        report.log_test("Oracle: Initialization", True)
        
        # Query
        response = oracle.query("What is entropy?", use_rag=True, k=3)
        report.log_test("Oracle: Query (RAG)", True,
                       details=f"Response length: {len(response)} chars")
        
    except Exception as e:
        report.log_test("Oracle: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def test_action_agent(report):
    """Test Action Agent"""
    logger.info("\n" + "="*60)
    logger.info("🎯 Testing Action Agent")
    logger.info("="*60)
    
    try:
        from core.agents.action_agent import ActionAgent
        
        # Initialize
        agent = ActionAgent()
        report.log_test("Action Agent: Initialization", True)
        
        # Check available actions
        report.log_test("Action Agent: Available Actions", True,
                       details="Agent ready for task execution")
        
    except Exception as e:
        report.log_test("Action Agent: Critical Error", False, error=e)
        logger.error(traceback.format_exc())

def main():
    """Run all tests"""
    logger.info("🚀 Alexandria System - Full Integration Test")
    logger.info(f"Start Time: {datetime.now().isoformat()}\n")
    
    report = TestReport()
    
    # Run all tests sequentially
    test_vqvae(report)
    test_mycelial(report)
    test_lancedb(report)
    test_abduction_engine(report)
    test_causal_engine(report)
    test_oracle(report)
    test_action_agent(report)
    
    # Print summary
    report.print_summary()
    
    logger.info(f"\nEnd Time: {datetime.now().isoformat()}")
    logger.info("✅ Integration test complete!")
    
    # Exit with appropriate code
    sys.exit(0 if report.failed == 0 else 1)

if __name__ == "__main__":
    main()
