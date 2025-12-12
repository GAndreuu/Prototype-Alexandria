import torch
import torch.nn as nn
from .layers import OrthogonalProductQuantizer

class MonolithV13(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=None):
        super().__init__()
        
        # Se latent_dim não especificado, usa input_dim (backward compatibility)
        latent_dim = latent_dim or input_dim
        
        # Encoder: Projeta o vetor sujo para o espaço latente
        # Arquitetura Bottleneck com suporte a latent_dim flexível
        # Default: (384 -> 256 -> 384) para modelo antigo
        # Wiki:    (384 -> 1024 -> 512) para modelo wiki-trained
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),  # Agora flexível
            nn.LayerNorm(latent_dim)
        )
        
        # O Quantizador (agora usa latent_dim)
        self.quantizer = OrthogonalProductQuantizer(
            num_heads=4, 
            embedding_dim=latent_dim,  # Flexível: 384D ou 512D
            num_embeddings=256  # 1 Byte por head
        )
        
        # Decoder: Reconstrói a partir dos códigos discretos
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # Agora flexível
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # Sempre retorna input_dim
        )

    def forward(self, x):
        # 1. Encode
        z = self.encoder(x)
        
        # 2. Quantize (O gargalo de informação)
        # z_q: vetor reconstruído aproximado
        # indices: os códigos inteiros (salvos em disco)
        z_q, indices, distances = self.quantizer(z)
        
        # 3. Decode
        out = self.decoder(z_q)
        
        return {
            "reconstructed": out,
            "indices": indices,
            "z_e": z,    # Latente pré-quantização (usado na loss)
            "z_q": z_q   # Latente pós-quantização (usado na loss)
        }
    
    def forward_with_head_mask(self, x, head_mask):
        """
        Forward pass com máscara por head (para ablação).
        
        Args:
            x: Input tensor [B, D]
            head_mask: Tensor [H] com 1=usa head, 0=zera head
        
        Returns:
            dict com 'reconstructed', 'indices', 'z_e', 'z_q', 'z_q_masked'
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices, distances = self.quantizer(z)
        
        # Reshape to separate heads
        B, D = z_q.shape
        H = self.quantizer.num_heads
        Hd = D // H
        z_q_heads = z_q.view(B, H, Hd)
        
        # Apply mask
        z_q_masked_heads = z_q_heads * head_mask.view(1, H, 1)
        z_q_masked = z_q_masked_heads.view(B, D)
        
        # Decode with masked z_q
        out = self.decoder(z_q_masked)
        
        return {
            "reconstructed": out,
            "indices": indices,
            "z_e": z,
            "z_q": z_q,
            "z_q_masked": z_q_masked
        }

