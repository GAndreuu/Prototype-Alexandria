"""
MonolithWiki - Exact architecture used for wiki training.
Different from MonolithV13 due to training script specificities.
"""
import torch
import torch.nn as nn

class ProductQuantizerSimple(nn.Module):
    """Simple PQ for inference (no EMA tracking)."""
    
    def __init__(self, num_heads=4, codebook_size=256, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.head_dim = head_dim
        
        self.codebooks = nn.Parameter(
            torch.randn(num_heads, codebook_size, head_dim)
        )
        
        # These are from training but not used in inference
        self.register_buffer('usage', torch.zeros(num_heads, codebook_size))
        self.register_buffer('steps_unused', torch.zeros(num_heads, codebook_size))
    
    def forward(self, z):
        B = z.shape[0]
        z_heads = z.view(B, self.num_heads, self.head_dim)
        
        z_q_parts = []
        indices_list = []
        
        for h in range(self.num_heads):
            z_h = z_heads[:, h, :]
            cb_h = self.codebooks[h]
            
            # Find nearest
            dist = torch.cdist(z_h, cb_h, p=2)
            idx = dist.argmin(dim=1)
            
            # Quantize
            z_q_h = cb_h[idx]
            
            # STE
            z_q_h = z_h + (z_q_h - z_h).detach()
            
            z_q_parts.append(z_q_h)
            indices_list.append(idx)
        
        z_q = torch.cat(z_q_parts, dim=1)
        indices = torch.stack(indices_list, dim=1)
        
        return z_q, indices, None


class MonolithWiki(nn.Module):
    """Exact architecture from wiki training (reverse-engineered from weights)."""
    
    def __init__(self, input_dim=384, hidden_dim=512):
        super().__init__()
        
        # Encoder: 384 → 1024 → 512
        # Weights show: encoder.0 (Linear), encoder.1 (LayerNorm), encoder.4 (Linear)
        # Layers 2,3 are GELU/Dropout or similar (no params saved)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),      # 0: [1024, 384]
            nn.LayerNorm(hidden_dim * 2),              # 1: [1024]
            nn.GELU(),                                  # 2: (no params)
            nn.Dropout(0.0),                            # 3: (no params, placeholder)
            nn.Linear(hidden_dim * 2, hidden_dim),     # 4: [512, 1024]
        )
        
        # Quantizer
        self.quantizer = ProductQuantizerSimple(
            num_heads=4,
            codebook_size=256,
            head_dim=hidden_dim // 4
        )
        
        # Decoder: 512 → 1024 → 384
        # Weights show: decoder.0 (Linear), decoder.1 (LayerNorm), decoder.4 (Linear)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),     # 0: [1024, 512]
            nn.LayerNorm(hidden_dim * 2),              # 1: [1024]
            nn.GELU(),                                  # 2: (no params)
            nn.Dropout(0.0),                            # 3: (no params, placeholder)
            nn.Linear(hidden_dim * 2, input_dim),      # 4: [384, 1024]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, _ = self.quantizer(z)
        x_recon = self.decoder(z_q)
        
        return {
            'reconstructed': x_recon,
            'indices': indices,
            'z_e': z,
            'z_q': z_q
        }


if __name__ == "__main__":
    # Test
    model = MonolithWiki()
    print("MonolithWiki structure:")
    for k, v in model.state_dict().items():
        print(f"  {k}: {v.shape}")
