# ============================================
# MONOLITH V13 RETRAIN — Alexandria-Native
# ============================================
# Retreina o MonolithV13 usando a arquitetura EXATA do sistema
# com correções anti-colapso de codebook
# ============================================

import os
import sys
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly não disponível")

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TrainConfig:
    # Architecture (must match MonolithV13)
    input_dim: int = 384
    hidden_dim: int = 256
    latent_dim: int = 384  # Same as input for original model
    num_heads: int = 4
    codebook_size: int = 256
    head_dim: int = 96  # latent_dim // num_heads

    # Training
    lr: float = 3e-4
    min_lr: float = 1e-5
    batch_size: int = 256
    epochs: int = 25
    steps_per_epoch: int = 1500
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Loss weights
    recon_weight: float = 1.0
    commit_weight: float = 0.25
    entropy_weight: float = 0.15  # Anti-collapse

    # EMA for codebook (faster adaptation)
    ema_decay: float = 0.95
    ema_epsilon: float = 1e-5

    # Dead code reset
    reset_warmup: int = 2000
    reset_threshold: int = 300
    reset_temperature: float = 0.1

    # Data
    data_path: str = "data/training_embeddings.npy"
    shuffle_each_epoch: bool = True

    # Targets
    target_recon: float = 0.08
    target_coverage: float = 0.80
    target_similarity: float = 0.5

    # Export
    export_path: str = "data/monolith_v13_retrained.pth"


CONFIG = TrainConfig()


# ============================================================
# IMPROVED ORTHOGONAL PRODUCT QUANTIZER
# ============================================================

class ImprovedOrthogonalProductQuantizer(nn.Module):
    """
    Quantizador com correções anti-colapso.
    Mantém interface compatível com OrthogonalProductQuantizer original.
    """
    
    def __init__(self, num_heads=4, embedding_dim=384, num_embeddings=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_embeddings = num_embeddings
        self.head_dim = embedding_dim // num_heads
        self.logK = math.log(num_embeddings)

        assert embedding_dim % num_heads == 0

        # Codebook como Parameter (compatível com original)
        self.codebooks = nn.Parameter(self._init_diverse_codebook())
        
        # Buffers para tracking (não salvos como parâmetros treináveis)
        self.register_buffer("inited", torch.ones(1))  # Já inicializado
        self.register_buffer("ema_count", torch.zeros(num_heads, num_embeddings))
        self.register_buffer("ema_weight", torch.zeros(num_heads, num_embeddings, self.head_dim))
        self.register_buffer("usage", torch.zeros(num_heads, num_embeddings))
        self.register_buffer("steps_unused", torch.zeros(num_heads, num_embeddings))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

    def _init_diverse_codebook(self) -> torch.Tensor:
        """Initialize with diverse, orthogonal-ish codes."""
        codebooks = torch.randn(self.num_heads, self.num_embeddings, self.head_dim)
        # Normalize each code to unit sphere
        codebooks = F.normalize(codebooks, dim=2)
        return codebooks

    def forward(self, z):
        """
        Compatible interface with original OrthogonalProductQuantizer.
        
        Args:
            z: [Batch, Embedding_Dim]
        Returns:
            z_q: Quantized vectors [Batch, Embedding_Dim]
            indices: Code indices [Batch, Heads]
            distances: Distance tensor for monitoring
        """
        bsz, dim = z.shape
        
        # Reshape to heads: [Batch, Heads, Head_Dim]
        z_reshaped = z.view(bsz, self.num_heads, self.head_dim)
        
        # Compute distances using cosine similarity (more stable)
        z_norm = F.normalize(z_reshaped, dim=2)  # [B, H, D]
        cb_norm = F.normalize(self.codebooks, dim=2)  # [H, K, D]
        
        # Similarity: [B, H, K]
        # For each head, compute similarity between each sample and all codes
        similarities = torch.einsum('bhd,hkd->bhk', z_norm, cb_norm)
        
        # Convert to distances (for compatibility)
        distances = 1.0 - similarities
        
        # Argmin (most similar = highest similarity = lowest distance)
        encoding_indices = similarities.argmax(dim=-1)  # [B, H]
        
        # Lookup quantized vectors
        z_q = torch.zeros_like(z_reshaped)
        for h in range(self.num_heads):
            z_q[:, h, :] = self.codebooks[h][encoding_indices[:, h]]
        
        # Straight-Through Estimator
        z_q = z_reshaped + (z_q - z_reshaped).detach()
        
        return z_q.view(bsz, dim), encoding_indices, distances

    def compute_entropy_loss(self, z):
        """
        Compute entropy-based loss to encourage uniform code usage.
        """
        bsz, dim = z.shape
        z_reshaped = z.view(bsz, self.num_heads, self.head_dim)
        
        z_norm = F.normalize(z_reshaped, dim=2)
        cb_norm = F.normalize(self.codebooks, dim=2)
        
        # Soft assignments via softmax over similarities
        similarities = torch.einsum('bhd,hkd->bhk', z_norm, cb_norm)
        soft_assign = F.softmax(similarities * 10.0, dim=-1)  # Temperature=0.1
        
        # Average assignment probabilities across batch
        avg_assign = soft_assign.mean(dim=0)  # [H, K]
        
        # Entropy per head
        entropy_per_head = -(avg_assign * (avg_assign + 1e-10).log()).sum(dim=1)
        max_entropy = math.log(self.num_embeddings)
        
        # Loss: penalize low entropy (want entropy close to max)
        entropy_loss = (1.0 - entropy_per_head / max_entropy).mean()
        
        return entropy_loss

    @torch.no_grad()
    def update_ema(self, z, indices, decay=0.95):
        """Update codebook via EMA (called after backward)."""
        self.total_steps += 1
        
        bsz = z.shape[0]
        z_reshaped = z.view(bsz, self.num_heads, self.head_dim)
        z_norm = F.normalize(z_reshaped, dim=2)
        
        for h in range(self.num_heads):
            idx_h = indices[:, h]
            z_h = z_norm[:, h]
            
            one_hot = F.one_hot(idx_h, self.num_embeddings).float()
            count = one_hot.sum(dim=0)
            weight_sum = one_hot.T @ z_h
            
            self.ema_count[h] = decay * self.ema_count[h] + (1 - decay) * count
            self.ema_weight[h] = decay * self.ema_weight[h] + (1 - decay) * weight_sum
            
            # Update codebook
            n = self.ema_count[h].unsqueeze(1) + 1e-5
            new_codes = self.ema_weight[h] / n
            new_codes = F.normalize(new_codes, dim=1)
            
            # Blend with learned codes
            self.codebooks.data[h] = 0.1 * self.codebooks.data[h] + 0.9 * new_codes
            self.codebooks.data[h] = F.normalize(self.codebooks.data[h], dim=1)
            
            # Track usage
            counts = torch.bincount(idx_h, minlength=self.num_embeddings).float()
            self.usage[h] += counts
            used = (counts > 0).nonzero().squeeze(-1)
            self.steps_unused[h] += 1
            if used.numel() > 0:
                self.steps_unused[h, used] = 0

    @torch.no_grad()
    def reset_dead_codes(self, z, indices, warmup=2000, threshold=300):
        """Reset unused codes to high-error samples."""
        if self.total_steps.item() < warmup:
            return 0
        
        bsz = z.shape[0]
        z_reshaped = z.view(bsz, self.num_heads, self.head_dim)
        total_reset = 0
        
        for h in range(self.num_heads):
            dead = self.steps_unused[h] >= threshold
            num_dead = int(dead.sum().item())
            if num_dead <= 0:
                continue
            
            # Find samples with highest quantization error
            z_h = z_reshaped[:, h]
            zq_h = self.codebooks[h][indices[:, h]]
            errors = ((z_h - zq_h) ** 2).sum(dim=1)
            
            _, hard_idx = errors.topk(min(num_dead, bsz))
            dead_idx = dead.nonzero().squeeze(-1)[:len(hard_idx)]
            
            for k, code in enumerate(dead_idx):
                # Replace with high-error sample + noise
                new_code = F.normalize(z_h[hard_idx[k:k+1]], dim=1).squeeze(0)
                self.codebooks.data[h, code] = new_code
                self.ema_weight[h, code] = new_code
                self.ema_count[h, code] = 1.0
                self.steps_unused[h, code] = 0
                total_reset += 1
        
        return total_reset

    def get_codes_from_indices(self, indices):
        """Reconstruct vectors from indices (compatibility method)."""
        bsz = indices.shape[0]
        z_q = torch.zeros(bsz, self.num_heads, self.head_dim, device=indices.device)
        for h in range(self.num_heads):
            z_q[:, h, :] = self.codebooks[h][indices[:, h]]
        return z_q.view(bsz, -1)

    def get_health_metrics(self, indices=None):
        """Get codebook health metrics."""
        # Code diversity (cosine similarity)
        avg_sim = {}
        for h in range(self.num_heads):
            codes = F.normalize(self.codebooks[h], dim=1)
            sim_matrix = codes @ codes.T
            mask = torch.eye(self.num_embeddings, device=codes.device, dtype=bool)
            sim_matrix.masked_fill_(mask, 0)
            avg_sim[h] = float(sim_matrix.sum() / (self.num_embeddings * (self.num_embeddings - 1)))
        
        # Usage
        util = {}
        for h in range(self.num_heads):
            used = (self.usage[h] > 0).sum().item()
            util[h] = used / self.num_embeddings
        
        return {
            "similarity": avg_sim,
            "avg_similarity": float(np.mean(list(avg_sim.values()))),
            "utilization": util,
            "avg_util": float(np.mean(list(util.values()))),
        }


# ============================================================
# MONOLITH V13 (Exact Architecture)
# ============================================================

class MonolithV13Retrain(nn.Module):
    """
    Exact MonolithV13 architecture with improved quantizer.
    """
    
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=None):
        super().__init__()
        latent_dim = latent_dim or input_dim
        
        # Encoder: Projects to latent space (EXACT same as original)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Quantizer (improved but compatible interface)
        self.quantizer = ImprovedOrthogonalProductQuantizer(
            num_heads=4,
            embedding_dim=latent_dim,
            num_embeddings=256
        )
        
        # Decoder: Reconstructs from quantized (EXACT same as original)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices, distances = self.quantizer(z)
        
        # Decode
        out = self.decoder(z_q)
        
        # Compute losses
        recon_loss = F.mse_loss(out, x)
        commit_loss = F.mse_loss(z, z_q.detach())
        entropy_loss = self.quantizer.compute_entropy_loss(z)
        
        return {
            "reconstructed": out,
            "indices": indices,
            "z_e": z,
            "z_q": z_q,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "entropy_loss": entropy_loss,
            "distances": distances,
        }

    def encode(self, x):
        """Encode and return indices."""
        with torch.no_grad():
            z = self.encoder(x)
            _, indices, _ = self.quantizer(z)
        return indices

    def to_monolith_v13_state_dict(self):
        """
        Convert to state dict compatible with original MonolithV13.
        """
        state = self.state_dict()
        
        # The quantizer in original uses just 'codebooks' as Parameter
        # Our improved version has same structure, just need to ensure compatibility
        compatible_state = {}
        for k, v in state.items():
            # Map our keys to original keys
            if k.startswith("quantizer."):
                # Original has: quantizer.codebooks, quantizer.inited
                if k in ["quantizer.codebooks", "quantizer.inited"]:
                    compatible_state[k] = v
                # Skip EMA buffers (not in original)
            else:
                compatible_state[k] = v
        
        return compatible_state


# ============================================================
# DATA LOADER
# ============================================================

class LocalEmbeddingDataset:
    def __init__(self, data_path: str, batch_size: int = 256):
        print(f"\n📂 Carregando embeddings de {data_path}...")
        self.data = np.load(data_path)
        self.batch_size = batch_size
        self.n_samples = self.data.shape[0]
        self.indices = np.arange(self.n_samples)
        print(f"✅ {self.n_samples:,} embeddings de {self.data.shape[1]}D")
        
    def shuffle(self):
        np.random.shuffle(self.indices)
        
    def __iter__(self):
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size // 2:
                continue
            yield torch.tensor(self.data[batch_indices], dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples // self.batch_size


# ============================================================
# LR SCHEDULER
# ============================================================

def cosine_warmup_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# TRAINING
# ============================================================

def train(cfg: TrainConfig):
    print("\n🚀 Retreinando MonolithV13 com Correções Anti-Colapso")
    print(f"   Entropy weight: {cfg.entropy_weight}")
    print(f"   EMA decay: {cfg.ema_decay}")
    print("=" * 70)

    # Data
    dataset = LocalEmbeddingDataset(cfg.data_path, cfg.batch_size)
    
    # Model
    model = MonolithV13Retrain(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim
    ).to(device)
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    total_steps = cfg.epochs * cfg.steps_per_epoch
    global_step = 0
    
    history = {
        "loss": [], "recon": [], "commit": [], "entropy": [],
        "similarity": [], "util": [], "resets": []
    }
    
    best_sim = float("inf")
    best_state = None
    
    for epoch in range(cfg.epochs):
        model.train()
        
        if cfg.shuffle_each_epoch:
            dataset.shuffle()
        
        losses, recons, commits, entropies, resets = [], [], [], [], []
        data_iter = iter(dataset)
        
        pbar = tqdm(range(min(cfg.steps_per_epoch, len(dataset))), 
                    desc=f"Ep {epoch+1}/{cfg.epochs}", leave=False)
        
        for _ in pbar:
            # LR schedule
            lr = cosine_warmup_lr(global_step, total_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            
            # Get batch
            try:
                x = next(data_iter).to(device)
            except StopIteration:
                dataset.shuffle()
                data_iter = iter(dataset)
                x = next(data_iter).to(device)
            
            # Forward
            opt.zero_grad(set_to_none=True)
            out = model(x)
            
            # Combined loss
            loss = (
                cfg.recon_weight * out["recon_loss"] +
                cfg.commit_weight * out["commit_loss"] +
                cfg.entropy_weight * out["entropy_loss"]
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            
            # EMA update (after gradient step)
            model.quantizer.update_ema(out["z_e"].detach(), out["indices"], cfg.ema_decay)
            
            # Reset dead codes
            n_reset = model.quantizer.reset_dead_codes(
                out["z_e"].detach(), out["indices"],
                cfg.reset_warmup, cfg.reset_threshold
            )
            
            # Track
            losses.append(float(loss.item()))
            recons.append(float(out["recon_loss"].item()))
            commits.append(float(out["commit_loss"].item()))
            entropies.append(float(out["entropy_loss"].item()))
            resets.append(n_reset)
            
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{out['recon_loss'].item():.4f}",
                "ent": f"{out['entropy_loss'].item():.3f}",
            })
        
        # Epoch metrics
        avg_loss = np.mean(losses) if losses else float("nan")
        avg_recon = np.mean(recons) if recons else float("nan")
        avg_commit = np.mean(commits) if commits else float("nan")
        avg_entropy = np.mean(entropies) if entropies else float("nan")
        total_resets = sum(resets)
        
        # Codebook health
        metrics = model.quantizer.get_health_metrics()
        
        history["loss"].append(float(avg_loss))
        history["recon"].append(float(avg_recon))
        history["commit"].append(float(avg_commit))
        history["entropy"].append(float(avg_entropy))
        history["similarity"].append(metrics["avg_similarity"])
        history["util"].append(metrics["avg_util"])
        history["resets"].append(float(total_resets))
        
        # Status
        s_recon = "✅" if avg_recon < cfg.target_recon else "🔴"
        s_sim = "✅" if metrics["avg_similarity"] < cfg.target_similarity else ("🟡" if metrics["avg_similarity"] < 0.8 else "🔴")
        
        print(
            f"Epoch {epoch+1:3d} │ "
            f"Loss {avg_loss:.4f} │ "
            f"Recon {avg_recon:.4f} {s_recon} │ "
            f"Sim {metrics['avg_similarity']:.3f} {s_sim} │ "
            f"Util {metrics['avg_util']:.1%} │ "
            f"Resets {total_resets}"
        )
        
        if (epoch + 1) % 5 == 0:
            print(f"         Per-head sim: {metrics['similarity']}")
            print(f"         Per-head util: {metrics['utilization']}")
        
        # Track best (by similarity - lower is better)
        if metrics["avg_similarity"] < best_sim:
            best_sim = metrics["avg_similarity"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state is not None and best_sim < 0.7:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n🏆 Restaurado BEST por similarity: {best_sim:.3f}")
    
    return model, history


# ============================================================
# EXPORT
# ============================================================

def export_model(model: MonolithV13Retrain, history: Dict, cfg: TrainConfig):
    print("\n📦 Exportando modelo...")
    
    # Get compatible state dict
    compatible_state = model.to_monolith_v13_state_dict()
    
    # Save in same format as original
    torch.save({
        "model_state_dict": compatible_state,
        "config": asdict(cfg),
        "history": history,
    }, cfg.export_path)
    
    print(f"✅ Modelo salvo em: {cfg.export_path}")
    
    # Verify
    print("\n🔍 Verificando compatibilidade...")
    checkpoint = torch.load(cfg.export_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    
    cb = state.get("quantizer.codebooks")
    if cb is not None:
        print(f"   Codebook shape: {cb.shape}")
        
        # Health check
        for h in range(4):
            codes = F.normalize(cb[h], dim=1)
            sim_matrix = codes @ codes.T
            mask = torch.eye(256, dtype=bool)
            sim_matrix.masked_fill_(mask, 0)
            avg_sim = float(sim_matrix.sum() / (256 * 255))
            status = "✅" if avg_sim < 0.5 else ("🟡" if avg_sim < 0.8 else "🔴")
            print(f"   Head {h} similarity: {avg_sim:.3f} {status}")
    
    # Plot
    if PLOTLY_AVAILABLE:
        fig = make_subplots(rows=2, cols=2, subplot_titles=["Loss", "Recon", "Similarity", "Utilization"])
        fig.add_trace(go.Scatter(y=history["loss"], name="Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(y=history["recon"], name="Recon"), row=1, col=2)
        fig.add_hline(y=cfg.target_recon, line_dash="dash", row=1, col=2)
        fig.add_trace(go.Scatter(y=history["similarity"], name="Sim"), row=2, col=1)
        fig.add_hline(y=cfg.target_similarity, line_dash="dash", row=2, col=1)
        fig.add_trace(go.Scatter(y=history["util"], name="Util"), row=2, col=2)
        fig.update_layout(height=600, title_text="MonolithV13 Retrain")
        plot_path = cfg.export_path.replace(".pth", "_training.html")
        fig.write_html(plot_path)
        print(f"📈 Gráficos: {plot_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MONOLITH V13 RETRAIN — Anti-Collapse Edition")
    print("=" * 70)
    
    model, history = train(CONFIG)
    export_model(model, history, CONFIG)
    
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(f"   Recon final:      {history['recon'][-1]:.4f}")
    print(f"   Similarity final: {history['similarity'][-1]:.3f}")
    print(f"   Utilization final: {history['util'][-1]:.1%}")
    print(f"   Modelo: {CONFIG.export_path}")
    print("=" * 70)
    print("\n✅ Treinamento completo!")
