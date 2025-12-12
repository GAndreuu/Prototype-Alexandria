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


def compute_head_balance_loss(z_q, num_heads: int):
    """
    Penaliza desequilíbrio extremo de energia (norma) entre heads.
    
    z_q      : [B, D] (concat de todas as heads)
    num_heads: número de heads (4 no Alexandria)
    
    Retorna um escalar pequeno se cada head carrega fração de energia parecida.
    """
    B, D = z_q.shape
    head_dim = D // num_heads
    z_q_heads = z_q.view(B, num_heads, head_dim)            # [B, H, Hd]
    
    # Energia média por head (norma² média)
    head_energy = (z_q_heads ** 2).sum(dim=-1).mean(dim=0)  # [H]
    head_energy = head_energy + 1e-8
    head_energy_norm = head_energy / head_energy.sum()      # fração de energia por head
    
    # Target = energia igualmente distribuída entre heads
    target = torch.full_like(head_energy_norm, 1.0 / num_heads)
    
    balance_loss = ((head_energy_norm - target) ** 2).mean()
    return balance_loss


def compute_code_usage_entropy_loss(indices, num_embeddings: int, target_entropy_frac: float = 0.8):
    """
    Incentiva cada head a usar um número razoável de códigos (alta entropia).
    
    indices           : [B, H] inteiros 0..num_embeddings-1
    num_embeddings    : 256 no Alexandria
    target_entropy_frac: fração da entropia máxima que queremos (ex.: 0.8 = 80%)
    
    Se uma head colapsa em poucos códigos, sua entropia cai e gera penalidade.
    """
    device = indices.device
    B, H = indices.shape
    entropies_norm = []

    for h in range(H):
        # Histograma de uso por código nessa head
        hist = torch.bincount(indices[:, h], minlength=num_embeddings).float()  # [N]
        p = hist / (hist.sum() + 1e-8)
        mask = p > 0
        p = p[mask]
        if p.numel() == 0:
            # Head completamente morta -> entropia zero
            entropies_norm.append(torch.tensor(0.0, device=device))
            continue

        H_bits = -(p * torch.log2(p)).sum()
        H_max = torch.log2(torch.tensor(float(num_embeddings), device=device))
        entropies_norm.append(H_bits / (H_max + 1e-8))

    entropies_norm = torch.stack(entropies_norm)  # [H]

    # Penaliza heads com entropia < target_entropy_frac
    penalty = torch.clamp(target_entropy_frac - entropies_norm, min=0.0)
    return penalty.mean()

