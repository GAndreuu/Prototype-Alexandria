"""
Monolith Encoder - Alexandria Integration
Inference-only version of the trained VQ-VAE model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonolithConfig:
    """Configuration matching training."""
    input_dim: int = 384
    hidden_dim: int = 512
    num_heads: int = 4
    codebook_size: int = 256
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads
    
    @classmethod
    def from_json(cls, path: str) -> 'MonolithConfig':
        with open(path) as f:
            data = json.load(f)
        return cls(
            input_dim=data.get('input_dim', 384),
            hidden_dim=data.get('hidden_dim', 512),
            num_heads=data.get('num_heads', 4),
            codebook_size=data.get('codebook_size', 256),
        )


class ProductQuantizerInference(nn.Module):
    """Product Quantizer for inference only."""
    
    def __init__(self, config: MonolithConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.codebook_size = config.codebook_size
        self.head_dim = config.head_dim
        
        self.register_buffer(
            'codebooks',
            torch.zeros(config.num_heads, config.codebook_size, config.head_dim)
        )
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize to indices."""
        B = z.shape[0]
        z_heads = z.view(B, self.num_heads, self.head_dim)
        
        indices_list = []
        for h in range(self.num_heads):
            z_h = z_heads[:, h, :]
            cb_h = self.codebooks[h]
            
            z_sq = (z_h ** 2).sum(dim=1, keepdim=True)
            c_sq = (cb_h ** 2).sum(dim=1).unsqueeze(0)
            zc = z_h @ cb_h.T
            
            dist = z_sq + c_sq - 2 * zc
            idx = dist.argmin(dim=1)
            indices_list.append(idx)
        
        return torch.stack(indices_list, dim=1)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct from indices."""
        B = indices.shape[0]
        
        z_q_parts = []
        for h in range(self.num_heads):
            codes = self.codebooks[h][indices[:, h]]
            z_q_parts.append(codes)
        
        return torch.cat(z_q_parts, dim=1)


class MonolithEncoder(nn.Module):
    """Monolith Encoder for inference."""
    
    def __init__(self, config: MonolithConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        
        # Quantizer
        self.quantizer = ProductQuantizerInference(config)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.input_dim),
        )
    
    @classmethod
    def load(cls, export_dir: str, device: str = 'cpu') -> 'MonolithEncoder':
        """Load trained model."""
        export_path = Path(export_dir)
        
        # Load config
        config_path = export_path / 'config.json'
        if config_path.exists():
            config = MonolithConfig.from_json(str(config_path))
        else:
            logger.warning("config.json not found, using defaults")
            config = MonolithConfig()
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = export_path / 'weights.pth'
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning(f"weights.pth not found in {export_dir}")
        
        model.to(device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def encode(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Encode embeddings to indices."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        x = x.to(next(self.parameters()).device)
        
        z = self.encoder(x)
        indices = self.quantizer.encode(z)
        
        if squeeze:
            indices = indices.squeeze(0)
        
        return indices
    
    @torch.no_grad()
    def decode(self, indices: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Decode indices to embeddings."""
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long)
        
        squeeze = False
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
            squeeze = True
        
        indices = indices.to(next(self.parameters()).device)
        
        z_q = self.quantizer.decode(indices)
        x_recon = self.decoder(z_q)
        
        if squeeze:
            x_recon = x_recon.squeeze(0)
        
        return x_recon


# ============================================================================
# WRAPPER FOR ALEXANDRIA
# ============================================================================

class MonolithAlexandriaWrapper:
    """
    Wrapper integrating Monolith with Alexandria.
    Replaces MycelialVQVAE.
    """
    
    def __init__(self, monolith: MonolithEncoder, mycelial_config: Optional = None):
        self.monolith = monolith
        
        try:
            from core.reasoning.mycelial_reasoning import MycelialReasoning
            self.mycelial = MycelialReasoning(mycelial_config)
        except ImportError:
            logger.warning("MycelialReasoning not found")
            self.mycelial = None
    
    @classmethod
    def load_default(cls, model_dir: str = "monolith_export", device: str = None) -> 'MonolithAlexandriaWrapper':
        """Load default wrapper."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        monolith = MonolithEncoder.load(model_dir, device)
        return cls(monolith)
    
    def encode(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Encode to indices (numpy)."""
        indices = self.monolith.encode(x)
        return indices.cpu().numpy()
    
    def decode(self, indices: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Decode from indices (numpy)."""
        recon = self.monolith.decode(indices)
        return recon.cpu().numpy()
    
    def observe(self, indices: Union[np.ndarray, list]) -> None:
        """Hebbian learning."""
        if self.mycelial is not None:
            self.mycelial.observe(indices)
    
    def observe_batch(self, indices_batch: np.ndarray) -> None:
        """Observe multiple."""
        if self.mycelial is not None:
            self.mycelial.observe_batch(indices_batch)
    
    def propagate(self, indices: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        """Propagate activation."""
        if self.mycelial is not None:
            return self.mycelial.propagate(indices, steps)
        else:
            activation = np.zeros((4, 256))
            for h, idx in enumerate(indices):
                activation[h, int(idx)] = 1.0
            return activation
    
    def synthesize(self, activation: np.ndarray) -> np.ndarray:
        """Convert activation to indices."""
        if self.mycelial is not None:
            return self.mycelial.synthesize(activation)
        else:
            return np.argmax(activation, axis=1)
    
    def reason(self, indices: np.ndarray, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate + synthesize."""
        if self.mycelial is not None:
            return self.mycelial.reason(indices, steps)
        else:
            activation = self.propagate(indices, steps)
            new_indices = self.synthesize(activation)
            return new_indices, activation
    
    def full_pipeline(self, embedding: np.ndarray, reason: bool = True) -> Dict:
        """Complete pipeline."""
        indices = self.encode(embedding)
        self.observe(indices)
        
        result = {
            'original_indices': indices,
            'original_embedding': embedding,
        }
        
        if reason and self.mycelial is not None:
            new_indices, activation = self.reason(indices)
            result['reasoned_indices'] = new_indices
            result['activation_pattern'] = activation
            result['reasoned_embedding'] = self.decode(new_indices)
        
        return result
    
    def get_network_stats(self) -> Dict:
        """Mycelial stats."""
        if self.mycelial is not None:
            return self.mycelial.get_network_stats()
        return {}
    
    def save_state(self, path: str = None) -> None:
        """Save Mycelial state."""
        if self.mycelial is not None:
            self.mycelial.save_state(path)
    
    def get_codebooks(self) -> np.ndarray:
        """Get codebooks."""
        return self.monolith.quantizer.codebooks.cpu().numpy()


# ============================================================================
# TESTING
# ============================================================================

def test_integration():
    """Test integration."""
    print("Testing Monolith integration...")
    
    try:
        wrapper = MonolithAlexandriaWrapper.load_default("monolith_export/")
        
        # Test encode/decode
        x = np.random.randn(384).astype(np.float32)
        indices = wrapper.encode(x)
        recon = wrapper.decode(indices)
        
        print(f"✅ encode/decode: {x.shape} → {indices.shape} → {recon.shape}")
        
        # Test observe
        wrapper.observe(indices)
        print(f"✅ observe works")
        
        # Test reason
        new_indices, activation = wrapper.reason(indices)
        print(f"✅ reason: {indices} → {new_indices}")
        
        print("✅ Integration working!")
        
    except FileNotFoundError:
        print("⚠️  monolith_export/ not found. Train model first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-integration":
        test_integration()
    else:
        print(__doc__)
