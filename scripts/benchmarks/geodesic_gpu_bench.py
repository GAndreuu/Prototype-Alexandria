
import time
import timeit
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bench_results.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeodesicBench")

# =============================================================================
# 1. TORCH IMPLEMENTATION (GPU)
# =============================================================================

class TorchMetric:
    """Implementação da Métrica Riemanniana usando PyTorch e Autograd (GPU)."""
    
    def __init__(self, dim: int, device: str = "cuda"):
        self.dim = dim
        self.device = device
        
        self.deformations_centers = torch.empty((0, dim), device=device)
        self.deformations_intensities = torch.empty((0,), device=device)
        self.deformations_radii = torch.empty((0,), device=device)
        
    def add_deformation(self, center: np.ndarray, intensity: float, radius: float):
        c = torch.tensor(center, dtype=torch.float32, device=self.device).unsqueeze(0)
        i = torch.tensor([intensity], dtype=torch.float32, device=self.device)
        r = torch.tensor([radius], dtype=torch.float32, device=self.device)
        
        self.deformations_centers = torch.cat([self.deformations_centers, c])
        self.deformations_intensities = torch.cat([self.deformations_intensities, i])
        self.deformations_radii = torch.cat([self.deformations_radii, r])
        
    def metric_matrix(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.eye(self.dim, device=self.device)
        if self.deformations_centers.shape[0] == 0:
            return g
            
        diffs = x.unsqueeze(0) - self.deformations_centers 
        dists_sq = torch.sum(diffs**2, dim=1)
        dists = torch.sqrt(dists_sq + 1e-8)
        
        weights = self.deformations_intensities * torch.exp(
            -0.5 * (dists / self.deformations_radii)**2
        )
        
        directions = diffs / (dists.unsqueeze(1) + 1e-8)
        outers = torch.bmm(directions.unsqueeze(2), directions.unsqueeze(1))
        deformations = torch.sum(outers * weights.view(-1, 1, 1), dim=0)
        
        return g + deformations

    def christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        g = self.metric_matrix(x) 
        g_inv = torch.linalg.inv(g)
        
        dim = self.dim
        dG = torch.zeros((dim, dim, dim), device=self.device)
        
        # Otimização possível: usar jacobian funcional, mas loop é mais seguro de implementar agora
        for i in range(dim):
            for j in range(dim):
                grad_g_ij = torch.autograd.grad(
                    g[i, j], x, create_graph=True, retain_graph=True
                )[0]
                dG[:, i, j] = grad_g_ij
        
        gamma = torch.zeros((dim, dim, dim), device=self.device)
        for k in range(dim):
            term = 0.5 * torch.einsum(
                'l,ijl->ij', 
                g_inv[k, :], 
                (dG.permute(1, 2, 0) + dG.permute(2, 1, 0) - dG)
            )
            gamma[k] = term
            
        return gamma

    def geodesic_step(self, x: torch.Tensor, v: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.christoffel_symbols(x)
        v_v = torch.outer(v, v)
        accel = -1.0 * torch.sum(gamma * v_v.unsqueeze(0), dim=(1, 2))
        v_new = v + accel * dt
        x_new = x + v_new * dt
        return x_new, v_new

# =============================================================================
# 2. NUMPY IMPLEMENTATION (CPU)
# =============================================================================

class NumpyMetric:
    """Implementação da Métrica Riemanniana usando Numpy (CPU)."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.deformations_centers = np.empty((0, dim))
        self.deformations_intensities = np.empty((0,))
        self.deformations_radii = np.empty((0,))
        
    def add_deformation(self, center: np.ndarray, intensity: float, radius: float):
        self.deformations_centers = np.vstack([self.deformations_centers, center]) if len(self.deformations_centers) > 0 else np.array([center])
        self.deformations_intensities = np.append(self.deformations_intensities, intensity)
        self.deformations_radii = np.append(self.deformations_radii, radius)
        
    def metric_matrix(self, x: np.ndarray) -> np.ndarray:
        g = np.eye(self.dim)
        if len(self.deformations_centers) == 0:
            return g
            
        diffs = x - self.deformations_centers
        dists_sq = np.sum(diffs**2, axis=1)
        dists = np.sqrt(dists_sq + 1e-8)
        
        weights = self.deformations_intensities * np.exp(
            -0.5 * (dists / self.deformations_radii)**2
        )
        
        directions = diffs / (dists[:, np.newaxis] + 1e-8)
        
        # Manual outer product sum
        for k in range(len(weights)):
            d = directions[k]
            outer = np.outer(d, d)
            g += weights[k] * outer
            
        return g

    def christoffel_symbols(self, x: np.ndarray, epsilon=1e-5) -> np.ndarray:
        # Diferenças finitas para CPU
        dim = self.dim
        gamma = np.zeros((dim, dim, dim))
        g_inv = np.linalg.inv(self.metric_matrix(x))
        
        # dG[k, i, j] = ∂_k g_ij
        dG = np.zeros((dim, dim, dim))
        
        for k in range(dim):
            x_plus = x.copy(); x_plus[k] += epsilon
            x_minus = x.copy(); x_minus[k] -= epsilon
            g_plus = self.metric_matrix(x_plus)
            g_minus = self.metric_matrix(x_minus)
            dG[k] = (g_plus - g_minus) / (2 * epsilon)
            
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    # Γ^k_ij = 0.5 * g^kl * (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
                    val = 0.0
                    for l in range(dim):
                        term = dG[i, j, l] + dG[j, i, l] - dG[l, i, j]
                        val += g_inv[k, l] * term
                    gamma[k, i, j] = 0.5 * val
        return gamma

    def geodesic_step(self, x: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        gamma = self.christoffel_symbols(x)
        dim = self.dim
        accel = np.zeros(dim)
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    accel[k] -= gamma[k, i, j] * v[i] * v[j]
                    
        v_new = v + accel * dt
        x_new = x + v_new * dt
        return x_new, v_new

# =============================================================================
# 3. BENCHMARK RUNNER
# =============================================================================

def run_cpu_benchmark(dim=32, steps=100):
    logger.info(f"Iniciando CPU Benchmark {dim}d...")
    nm = NumpyMetric(dim)
    nm.add_deformation(np.zeros(dim), 0.5, 1.0)
    
    x = np.ones(dim) * 0.5
    v = np.zeros(dim)
    v[0] = 1.0
    
    t0 = time.time()
    for _ in range(steps):
        x, v = nm.geodesic_step(x, v, 0.01)
    t1 = time.time()
    
    elapsed = t1 - t0
    logger.info(f"CPU {dim}d: {elapsed:.4f}s")
    return steps / elapsed

def run_gpu_benchmark(dim=32, steps=100, device="cuda"):
    logger.info(f"Iniciando GPU Benchmark {dim}d...")
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available")
        return 0
        
    tm = TorchMetric(dim, device)
    tm.add_deformation(np.zeros(dim), 0.5, 1.0)
    
    x = torch.ones(dim, device=device) * 0.5
    v = torch.zeros(dim, device=device)
    v[0] = 1.0
    
    # Warmup
    tm.geodesic_step(x, v, 0.01)
    torch.cuda.synchronize() if device == "cuda" else None
    
    t0 = time.time()
    for _ in range(steps):
        x, v = tm.geodesic_step(x, v, 0.01)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()
    
    elapsed = t1 - t0
    logger.info(f"GPU {dim}d: {elapsed:.4f}s")
    return steps / elapsed

def main():
    print("="*60)
    print("🚀 GEODESIC FLOW BENCHMARK: CPU vs GPU (Standalone)")
    print("="*60)
    
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
    # --- 32D ---
    print("\n--- 32 Dimensions ---")
    cpu_fps = run_cpu_benchmark(dim=32, steps=20) # Menos steps pq CPU é lenta com diferencas finitas
    print(f"CPU (Numpy): {cpu_fps:.2f} steps/s")
    
    if gpu_available:
        gpu_fps = run_gpu_benchmark(dim=32, steps=20)
        print(f"GPU (Torch): {gpu_fps:.2f} steps/s")
        print(f"Speedup: {gpu_fps/cpu_fps:.2f}x")
    
    # --- 128D (Simulando expansão) ---
    print("\n--- 128 Dimensions ---")
    # CPU vai ser muito lenta aqui (O(d^5) para diferencas finitas na verdade?
    # Christoffel é O(d^3), derivativa é d*O(d^2)? = O(d^4)
    # CPU com 128d vai demorar muito. Steps=1
    cpu_fps_128 = run_cpu_benchmark(dim=128, steps=1) 
    print(f"CPU (Numpy): {cpu_fps_128:.2f} steps/s")
    
    if gpu_available:
        gpu_fps_128 = run_gpu_benchmark(dim=128, steps=20)
        print(f"GPU (Torch): {gpu_fps_128:.2f} steps/s")
        print(f"Speedup: {gpu_fps_128/cpu_fps_128:.2f}x")

if __name__ == "__main__":
    main()
