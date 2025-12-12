import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np

# Core Imports
from core.field.pre_structural_field import PreStructuralField, PreStructuralConfig
from config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlexandriaAPI")

# Initialize App
app = FastAPI(
    title="Alexandria Cognitive System",
    description="Neural Interface for the Pre-Structural Field",
    version="12.0"
)

# Global State
class SystemState:
    def __init__(self):
        self.field: Optional[PreStructuralField] = None

system = SystemState()

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Alexandria System...")
    
    # Initialize Core Field
    try:
        config = PreStructuralConfig(
            base_dim=settings.MANIFOLD_DIM,  # 32d optimized
            temperature=1.0
        )
        system.field = PreStructuralField(config)
        logger.info("PreStructuralField initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PreStructuralField: {e}")
        # Dont crash, allow API to start in limited mode
        
@app.get("/")
async def root():
    return {
        "system": "Alexandria",
        "status": "online",
        "field_initialized": system.field is not None
    }

@app.get("/health")
async def health_check():
    if system.field is None:
        return {"status": "degraded", "reason": "Field not initialized"}
    return {"status": "healthy"}

class TriggerRequest(BaseModel):
    embedding: List[float]
    intensity: float = 1.0

@app.post("/trigger")
async def trigger_concept(request: TriggerRequest):
    if system.field is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        emb_array = np.array(request.embedding)
        state = system.field.trigger(emb_array, intensity=request.intensity)
        return {
            "status": "triggered",
            "attractors": len(state.attractors),
            "free_energy": float(state.global_free_energy)
        }
    except Exception as e:
        logger.error(f"Trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
