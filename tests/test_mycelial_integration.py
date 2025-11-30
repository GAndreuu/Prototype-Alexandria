import sys
import os
from pathlib import Path
import logging

# Setup path
sys.path.append(os.getcwd())

import torch
import numpy as np
from core.mycelial_reasoning import MycelialReasoning
from core.neural_learner import V2Learner
from sentence_transformers import SentenceTransformer
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

def test_full_pipeline():
    logger.info("🚀 Starting Integration Test: Mycelial Reasoning Pipeline")

    # 1. Initialize Components
    logger.info("1. Initializing Components...")
    try:
        mycelial = MycelialReasoning()
        learner = V2Learner() # Loads MonolithV13
        encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("✅ Components initialized.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return

    # 2. Input Data
    input_text = "A inteligência artificial evolui através de padrões recursivos."
    logger.info(f"2. Input Text: '{input_text}'")

    # 3. Encode Text -> Vector
    logger.info("3. Encoding Text -> Vector...")
    try:
        embedding = encoder.encode([input_text])[0] # 384D
        logger.info(f"✅ Embedding shape: {embedding.shape}")
    except Exception as e:
        logger.error(f"❌ Text encoding failed: {e}")
        return

    # 4. Encode Vector -> Indices (Monolith)
    logger.info("4. Encoding Vector -> Indices (Monolith)...")
    try:
        with torch.no_grad():
            t_emb = torch.tensor([embedding], dtype=torch.float32).to(learner.device)
            out = learner.model(t_emb)
            indices = out['indices'].cpu().numpy().flatten()
        logger.info(f"✅ Indices: {indices}")
    except Exception as e:
        logger.error(f"❌ Monolith encoding failed: {e}")
        return

    # 5. Observe (Learn)
    logger.info("5. Observing (Hebbian Learning)...")
    try:
        mycelial.observe(indices)
        stats = mycelial.get_network_stats()
        logger.info(f"✅ Observation recorded. Total observations: {stats['total_observations']}")
    except Exception as e:
        logger.error(f"❌ Observation failed: {e}")
        return

    # 6. Reason (Propagate)
    logger.info("6. Reasoning (Propagation)...")
    try:
        new_indices, activation = mycelial.reason(indices, steps=5)
        logger.info(f"✅ Reasoned Indices: {new_indices}")
        logger.info(f"✅ Activation mean: {np.mean(activation):.6f}")
    except Exception as e:
        logger.error(f"❌ Reasoning failed: {e}")
        return

    logger.info("🎉 Integration Test PASSED!")

if __name__ == "__main__":
    test_full_pipeline()
