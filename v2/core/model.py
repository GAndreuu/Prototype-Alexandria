import torch
import torch.nn as nn
from .layers import OrthogonalProductQuantizer

class MonolithV13(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256):
        super().__init__()
        
        # Encoder: Projeta o vetor sujo para o espaço latente
        # Arquitetura Bottleneck (384 -> 256 -> 384) para forçar limpeza
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim), # Volta pra dim original para o VQ cortar
            nn.LayerNorm(input_dim)
        )
        
        # O Quantizador
        self.quantizer = OrthogonalProductQuantizer(
            num_heads=4, 
            embedding_dim=input_dim, 
            num_embeddings=256 # 1 Byte por head
        )
        
        # Decoder: Reconstrói a partir dos códigos discretos
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
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
