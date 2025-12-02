import torch
import torch.nn.functional as F

def compute_orthogonal_loss(quantizer_module):
    """
    Calcula a perda de ortogonalidade entre as cabeças do quantizador.
    O objetivo é garantir que a Head 1 aprenda coisas diferentes da Head 2.
    """
    codebooks = quantizer_module.codebooks # [Heads, Num_Emb, Head_Dim]
    num_heads = codebooks.shape[0]
    
    if num_heads < 2:
        return torch.tensor(0.0, device=codebooks.device)

    # Para representar a "direção" geral de uma Head, usamos a média dos seus vetores
    head_centers = torch.mean(codebooks, dim=1) # [Heads, Head_Dim]
    
    # Normalizamos para calcular Cosseno
    head_centers = F.normalize(head_centers, p=2, dim=1)
    
    # Matriz de Similaridade (Gram Matrix): Head vs Head
    similarity_matrix = torch.mm(head_centers, head_centers.t()) # [Heads, Heads]
    
    # Queremos que a matriz seja a Identidade (1 na diagonal, 0 no resto)
    identity = torch.eye(num_heads, device=codebooks.device)
    
    # Penalizamos qualquer valor fora da diagonal que seja diferente de 0
    ortho_loss = ((similarity_matrix - identity) ** 2).sum() / (num_heads * (num_heads - 1))
    
    return ortho_loss

def compute_vq_commitment_loss(z, z_q, beta=0.25):
    """
    Loss padrão do VQ-VAE.
    Args:
        z: Vetor original do encoder (antes da quantização)
        z_q: Vetor quantizado (após lookup)
        beta: Peso do compromisso (o quanto o codebook deve se mover)
    """
    # 1. O Codebook deve se mover para perto do Encoder
    codebook_loss = F.mse_loss(z_q, z.detach())
    
    # 2. O Encoder deve se mover para perto do Codebook
    commitment_loss = F.mse_loss(z_q.detach(), z)
    
    loss = codebook_loss + beta * commitment_loss
    return loss
