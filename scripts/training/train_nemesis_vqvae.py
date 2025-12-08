
import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import lancedb
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Handle Plotly for reporting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly/Scipy not found. Visual reports will be skipped.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

# =============================================================================
# CONFIGURAÇÃO V3 (NEMESIS ADAPTADA)
# =============================================================================

@dataclass
class ConfigV3:
    # Arquitetura
    input_dim: int = 384           # MiniLM output (LanceDB vectors)
    hidden_dim: int = 512          # Latent space
    num_heads: int = 4             # Product quantization heads
    codebook_size: int = 256       # Codes per head (256^4 combinations)
    
    # Loss weights
    commitment_cost: float = 0.25
    entropy_weight: float = 0.1
    orthogonal_weight: float = 0.5  # Penaliza correlação entre heads
    
    # Codebook management
    reset_threshold: int = 50      # Threshold para resetar códigos mortos
    temperature: float = 1.0
    ema_decay: float = 0.99         # EMA para codebooks
    
    # Training
    lr: float = 3e-4
    min_lr: float = 1e-5
    batch_size: int = 256           # Aumentei para CPU/Vetores (384 floats é leve)
    epochs: int = 5                 # Rápido, pois já temos vetores prontos
    steps_per_epoch: int = 1000     # Ajustável baseado no tamanho do dataset
    warmup_steps: int = 200
    
    # Dataset
    dataset_name: str = "Alexandria Local (LanceDB)"

    @property
    def head_dim(self):
        return self.hidden_dim // self.num_heads

CONFIG = ConfigV3()

# =============================================================================
# ARQUITETURA V3 (Code provided by User)
# =============================================================================

class ProductQuantizerV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.codebook_size = cfg.codebook_size
        self.head_dim = cfg.head_dim
        self.ema_decay = cfg.ema_decay
        
        self.head_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.head_dim * 2),
                nn.GELU(),
                nn.Linear(cfg.head_dim * 2, cfg.head_dim),
                nn.LayerNorm(cfg.head_dim)
            )
            for _ in range(cfg.num_heads)
        ])
        
        self.register_buffer('codebooks', torch.randn(cfg.num_heads, cfg.codebook_size, cfg.head_dim) * 0.1)
        self.register_buffer('codebook_ema_count', torch.zeros(cfg.num_heads, cfg.codebook_size))
        self.register_buffer('codebook_ema_weight', torch.randn(cfg.num_heads, cfg.codebook_size, cfg.head_dim) * 0.1)
        
        self.register_buffer('usage', torch.zeros(cfg.num_heads, cfg.codebook_size))
        self.register_buffer('steps_unused', torch.zeros(cfg.num_heads, cfg.codebook_size))
        self.register_buffer('total_steps', torch.tensor(0))
        
        self.head_decoders = nn.ModuleList([
            nn.Linear(cfg.head_dim, cfg.hidden_dim // cfg.num_heads)
            for _ in range(cfg.num_heads)
        ])

    def forward(self, z, training=True):
        B = z.shape[0]
        z_heads = []
        for h in range(self.num_heads):
            z_h = self.head_projections[h](z)
            z_heads.append(z_h)
        
        z_heads = torch.stack(z_heads, dim=1)
        
        all_indices = []
        all_dists = []
        z_q_parts = []
        
        for h in range(self.num_heads):
            z_h = z_heads[:, h, :]
            cb_h = self.codebooks[h]
            z_sq = (z_h ** 2).sum(dim=1, keepdim=True)
            c_sq = (cb_h ** 2).sum(dim=1, keepdim=True).T
            zc = z_h @ cb_h.T
            dist_h = z_sq + c_sq - 2 * zc
            idx_h = dist_h.argmin(dim=1)
            all_dists.append(dist_h)
            all_indices.append(idx_h)
            z_q_h = cb_h[idx_h]
            z_q_parts.append(z_q_h)
        
        indices = torch.stack(all_indices, dim=1)
        dists = torch.stack(all_dists, dim=1)
        
        decoded_parts = []
        for h in range(self.num_heads):
            decoded_parts.append(self.head_decoders[h](z_q_parts[h]))
        z_q = torch.cat(decoded_parts, dim=1)
        
        z_projected = torch.cat([self.head_decoders[h](z_heads[:, h]) for h in range(self.num_heads)], dim=1)
        z_q_st = z_projected + (z_q - z_projected).detach()
        
        commit = F.mse_loss(z_projected, z_q.detach())
        codebook = F.mse_loss(z_projected.detach(), z_q)
        
        probs = F.softmax(-dists * self.cfg.temperature, dim=-1)
        avg_probs = probs.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(dim=-1).mean()
        
        # FIX: Ensure log base matches codebook entropy logic if needed, but keeping original
        entropy_loss = 1 - entropy / np.log(self.codebook_size)
        
        ortho_loss = self._orthogonal_loss(indices)
        
        return {
            'z_q': z_q_st,
            'indices': indices,
            'commit': commit,
            'codebook': codebook,
            'entropy': entropy_loss,
            'orthogonal': ortho_loss,
            'z_heads': z_heads.detach()
        }
    
    def update_codebooks(self, z_heads, indices):
        self._update_codebooks_ema(z_heads, indices)
        self._track_usage(indices, z_heads.shape[0], z_heads)
    
    def _orthogonal_loss(self, indices):
        B = indices.shape[0]
        one_hots = [F.one_hot(indices[:, h], self.codebook_size).float() for h in range(self.num_heads)]
        ortho_loss = 0.0
        count = 0
        for i in range(self.num_heads):
            for j in range(i + 1, self.num_heads):
                co_occur = one_hots[i].T @ one_hots[j]
                co_occur = co_occur / (B + 1e-8)
                p_i = one_hots[i].mean(dim=0, keepdim=True)
                p_j = one_hots[j].mean(dim=0, keepdim=True).T
                expected = p_i.T @ p_j.T
                deviation = (co_occur - expected).abs().mean()
                ortho_loss += deviation
                count += 1
        return ortho_loss / count if count > 0 else 0.0
    
    def _update_codebooks_ema(self, z_heads, indices):
        with torch.no_grad():
            for h in range(self.num_heads):
                one_hot = F.one_hot(indices[:, h], self.codebook_size).float()
                count = one_hot.sum(dim=0)
                z_h = z_heads[:, h]
                dw = one_hot.T @ z_h
                self.codebook_ema_count[h] = self.ema_decay * self.codebook_ema_count[h] + (1 - self.ema_decay) * count
                self.codebook_ema_weight[h] = self.ema_decay * self.codebook_ema_weight[h] + (1 - self.ema_decay) * dw
                n = self.codebook_ema_count[h].unsqueeze(1) + 1e-5
                self.codebooks[h] = self.codebook_ema_weight[h] / n
    
    def _track_usage(self, indices, B, z_heads):
        with torch.no_grad():
            self.steps_unused += 1
            self.total_steps += 1
            for h in range(self.num_heads):
                uniqs, counts = indices[:, h].unique(return_counts=True)
                self.usage[h, uniqs] += counts.float()
                self.steps_unused[h, uniqs] = 0
                dead = self.steps_unused[h] > self.cfg.reset_threshold
                num_dead = dead.sum().item()
                if num_dead > 0 and self.total_steps > 100:
                    rand_idx = torch.randint(0, B, (num_dead,), device=z_heads.device)
                    self.codebooks[h, dead] = z_heads[rand_idx, h].clone()
                    self.codebook_ema_weight[h, dead] = z_heads[rand_idx, h].clone()
                    self.codebook_ema_count[h, dead] = 1.0
                    self.steps_unused[h, dead] = 0

    def utilization(self):
        u = self.usage.cpu().numpy()
        return {h: float((u[h] > 0).sum() / self.codebook_size) for h in range(self.num_heads)}
    
    def get_head_correlation(self, indices):
        correlations = {}
        for i in range(self.num_heads):
            for j in range(i + 1, self.num_heads):
                same = (indices[:, i] == indices[:, j]).float().mean()
                correlations[f'{i}-{j}'] = same.item()
        return correlations

class MonolithV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.quantizer = ProductQuantizerV3(cfg)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        q = self.quantizer(z, self.training)
        recon = self.decoder(q['z_q'])
        recon_loss = F.mse_loss(recon, x)
        total = (recon_loss + self.cfg.commitment_cost * q['commit'] + q['codebook'] +
                 self.cfg.entropy_weight * q['entropy'] + self.cfg.orthogonal_weight * q['orthogonal'])
        return {
            'recon_loss': recon_loss, 'total_loss': total, 'entropy_loss': q['entropy'],
            'orthogonal_loss': q['orthogonal'], 'indices': q['indices']
        }
    
    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            q = self.quantizer(z, training=False)
            return q['indices']

# =============================================================================
# DATASET GENERATOR (LOCAL LANCE DB)
# =============================================================================

def local_data_generator(batch_size):
    """Streams existing vectors from LanceDB. Fastest possible training."""
    db_path = r"c:\Users\G\Desktop\Alexandria\data\lancedb_store"
    db = lancedb.connect(db_path)
    
    try:
        tbl = db.open_table(db.table_names()[0])
        print(f"✅ Training on Local LanceDB Table. Total Rows: {len(tbl)}")
    except IndexError:
        print("❌ No table found. Streaming random noise for testing.")
        while True:
            yield torch.randn(batch_size, 384)
            
    # Iterate indefinitely for training steps
    while True:
        # LanceDB 0.x iterator
        # We fetch vectors in larger chunks to minimize I/O overhead
        # then batches for yield
        
        # Using a generator query to stream
        iterator = tbl.search().limit(len(tbl)).to_arrow()
        
        # LanceDB returns Arrow Table
        # We can treat it as a stream
        batch_accum = []
        
        # Re-querying or just iterating?
        # For simplicity in this script, we'll load indices or just stream efficiently.
        # Given 193k rows, it fits in RAM (193k * 384 * 4 bytes ~= 300MB).
        # Loading ALL into memory is actually efficient and safe here.
        
        print("📥 Loading ALL vectors into RAM for high-speed training...")
        # Select only vector column
        vecs = tbl.search().limit(len(tbl)).select(['vector']).to_pandas()['vector']
        
        # Convert to numpy array
        # This list comprehension is fast enough for 200k
        data_matrix = np.stack(vecs.values).astype(np.float32)
        print(f"   Shape: {data_matrix.shape} | Size: {data_matrix.nbytes / 1024**2:.2f} MB")
        
        # Shuffle loop
        indices = np.arange(len(data_matrix))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(data_matrix), batch_size):
            end_idx = min(start_idx + batch_size, len(data_matrix))
            batch_indices = indices[start_idx:end_idx]
            
            # If batch is too small (end of data), drop or keep?
            # Creating torch tensor
            batch = torch.from_numpy(data_matrix[batch_indices])
            
            yield batch

# =============================================================================
# TRAINING LOOP
# =============================================================================

def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

def train_nemesis():
    print("\n🚀 [NEMESIS] Starting Training for 'Codebook Science'...")
    
    model = MonolithV3(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=1e-4)
    
    total_steps = CONFIG.epochs * CONFIG.steps_per_epoch
    gen = local_data_generator(CONFIG.batch_size)
    
    history = {'loss': [], 'util': []}
    global_step = 0
    
    try:
        for epoch in range(CONFIG.epochs):
            model.train()
            epoch_loss = []
            
            pbar = tqdm(range(CONFIG.steps_per_epoch), desc=f"Ep {epoch+1}/{CONFIG.epochs}")
            
            for _ in pbar:
                lr = get_lr(global_step, total_steps, CONFIG.warmup_steps, CONFIG.lr, CONFIG.min_lr)
                for pg in optimizer.param_groups: pg['lr'] = lr
                
                x = next(gen).to(device)
                if x.shape[0] < 2: continue
                
                optimizer.zero_grad()
                out = model(x)
                out['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # EMA Update
                with torch.no_grad():
                    z = model.encoder(x)
                    z_heads = torch.stack([
                        model.quantizer.head_projections[h](z) 
                        for h in range(CONFIG.num_heads)
                    ], dim=1)
                    model.quantizer.update_codebooks(z_heads, out['indices'])
                
                epoch_loss.append(out['total_loss'].item())
                global_step += 1
                
                pbar.set_postfix({'loss': f"{out['total_loss'].item():.4f}"})
            
            # Epoch End
            avg_loss = np.mean(epoch_loss)
            util = np.mean(list(model.quantizer.utilization().values()))
            history['loss'].append(avg_loss)
            history['util'].append(util)
            
            print(f"   Epoch {epoch+1} | Loss: {avg_loss:.4f} | Util: {util:.1%}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted (saving current state)...")
    
    # Save Model
    save_path = r"c:\Users\G\Desktop\Alexandria\data\monolith_v3_nemesis.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(CONFIG),
    }, save_path)
    print(f"\n✅ Model Saved: {save_path}")
    
    # Export Power-Law Report (if plotly)
    if PLOTLY_AVAILABLE:
        try:
            print("📊 Generating Power-Law Analysis...")
            usage = model.quantizer.usage.cpu().numpy()
            fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Head {i}' for i in range(4)])
            
            for h in range(4):
                counts = usage[h][usage[h] > 0]
                if len(counts) > 5:
                    counts_sorted = np.sort(counts)[::-1]
                    ranks = np.arange(1, len(counts_sorted) + 1)
                    
                    row, col = h // 2 + 1, h % 2 + 1
                    fig.add_trace(go.Scatter(
                        x=np.log10(ranks), y=np.log10(counts_sorted), mode='markers',
                        name=f'Head {h}'
                    ), row=row, col=col)
            
            report_path = r"c:\Users\G\Desktop\Alexandria\docs\reports\monolith_powerlaw.html"
            fig.write_html(report_path)
            print(f"   Report saved: {report_path}")
        except Exception as e:
            print(f"   Result generation error: {e}")

if __name__ == "__main__":
    train_nemesis()
