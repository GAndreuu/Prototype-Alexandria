import torch
import torch.nn as nn

class OrthogonalProductQuantizer(nn.Module):
    """
    Quantizador Vetorial de Produto (PQ) com Estimador Straight-Through.
    Divide o vetor em 'heads' e quantiza cada parte independentemente.
    """
    def __init__(self, num_heads=4, embedding_dim=384, num_embeddings=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_embeddings = num_embeddings
        self.head_dim = embedding_dim // num_heads

        assert embedding_dim % num_heads == 0, f"Embedding dim {embedding_dim} deve ser divisível por num_heads {num_heads}"

        # Codebooks: [Heads, Tamanho do Dicionário, Dimensão da Head]
        # Inicializamos com ruído pequeno para evitar colapso inicial
        self.codebooks = nn.Parameter(torch.randn(num_heads, num_embeddings, self.head_dim) * 0.02)

    def forward(self, z):
        """
        Args:
            z: Tensor de entrada [Batch, Embedding_Dim]
        Returns:
            z_q: Vetor quantizado (com gradiente habilitado via STE)
            indices: Índices inteiros dos códigos escolhidos [Batch, Heads]
            distances: Distâncias brutas para monitoramento
        """
        bsz, dim = z.shape
        
        # Reshape para separar as cabeças: [Batch, Heads, Head_Dim]
        z_reshaped = z.view(bsz, self.num_heads, self.head_dim)

        # --- Lógica de VQ (Vector Quantization) ---
        
        # Queremos calcular a distância ||z - c||^2 = z^2 + c^2 - 2zc
        # Expandimos dimensões para broadcast eficiente
        z_expanded = z_reshaped.unsqueeze(2)    # [Batch, Heads, 1, Head_Dim]
        codebooks_expanded = self.codebooks.unsqueeze(0) # [1, Heads, Num_Emb, Head_Dim]

        # Distância Euclidiana Quadrada
        # (z - c)^2
        # FIX: Using explicit permutation for correct broadcasting if implicit fails
        # But sticking to user provided code structure first.
        # Note: The user's code uses matmul(z_reshaped, codebooks.transpose).
        # z_reshaped: [B, H, D]
        # codebooks.T: [H, D, N]
        # PyTorch matmul might struggle with [B, H, D] x [H, D, N] broadcasting B vs H.
        # We will use the robust implementation we found earlier to ensure it works.
        
        # Termo 1: ||z||^2
        z_sq = torch.sum(z_expanded**2, dim=-1) # [B, H, 1]
        
        # Termo 2: ||c||^2
        c_sq = torch.sum(codebooks_expanded**2, dim=-1) # [1, H, N]
        
        # Termo 3: 2 * z * c^T
        # Robust implementation:
        # z: [B, H, D] -> [H, B, D]
        z_perm = z_reshaped.permute(1, 0, 2)
        # c: [H, N, D] -> [H, D, N]
        c_perm = self.codebooks.permute(0, 2, 1)
        # dot: [H, B, N]
        dot = torch.bmm(z_perm, c_perm)
        # dot: [B, H, N]
        dot = dot.permute(1, 0, 2)
        
        distances = z_sq + c_sq - 2 * dot
        # distances shape: [Batch, Heads, Num_Embeddings]

        # Encontrar o índice mais próximo (Argmin)
        encoding_indices = torch.argmin(distances, dim=-1) # [Batch, Heads]

        # --- Reconstrução (Lookup) ---
        # Cria um tensor vazio e preenche com os vetores escolhidos
        z_q = torch.zeros_like(z_reshaped)
        for h in range(self.num_heads):
            # Para cada head, pegamos os vetores correspondentes aos índices escolhidos
            z_q[:, h, :] = self.codebooks[h][encoding_indices[:, h]]

        # --- Straight-Through Estimator (STE) ---
        # O pulo do gato matemático.
        # Para o forward pass, usamos o valor quantizado (z_q).
        # Para o backward pass (gradiente), usamos o valor contínuo (z_reshaped).
        # Isso permite treinar a rede mesmo com a operação "argmin" que não é diferenciável.
        z_q = z_reshaped + (z_q - z_reshaped).detach()

        return z_q.view(bsz, dim), encoding_indices, distances

    def get_codes_from_indices(self, indices):
        """
        Reconstrói os vetores quantizados a partir dos índices.
        Args:
            indices: [Batch, Heads]
        Returns:
            z_q: [Batch, Embedding_Dim]
        """
        bsz = indices.shape[0]
        z_q = torch.zeros(bsz, self.num_heads, self.head_dim, device=indices.device)
        
        for h in range(self.num_heads):
            z_q[:, h, :] = self.codebooks[h][indices[:, h]]
            
        return z_q.view(bsz, -1)
