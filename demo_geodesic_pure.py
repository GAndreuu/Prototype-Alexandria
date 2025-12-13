#!/usr/bin/env python3
"""
demo_geodesic_pure.py

Demonstração "pura" de geodésicas em espaço curvo (Riemanniano),
sem depender de módulos do seu sistema.

Principais melhorias:
- Métrica anisotrópica vetorizada (rápida)
- Christoffel por diferenças finitas só em active_dims
- Integração semi-implícita + renormalização de energia (reduz drift)
- Shooting method (aproxima BVP: start->end) em vez de IVP puro
- Critérios de parada + "best-cut" quando não converge
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataclasses import dataclass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class DemoConfig:
    dim: int = 64
    num_attractors: int = 3
    seed: int = 42
    output_path: str = "geodesic_demo_output.png"


@dataclass(frozen=True)
class MetricConfig:
    attractor_strength: float = 5.0
    alpha: float = 0.1  # menor => influência mais longa
    eps_fd: float = 1e-4  # passo da diferença finita


@dataclass(frozen=True)
class FlowConfig:
    dt: float = 0.02
    max_steps: int = 200
    active_dims: int = 24  # Aumentado para melhor convergência
    energy_renorm: bool = True
    vel_damping: float = 0.0
    speed_floor: float = 1e-6

    # BVP (shooting)
    shooting_iters: int = 35
    lr: float = 0.35
    tol: float = 1e-2  # Relaxado para demo visual
    patience: int = 35


@dataclass
class SimpleGeodesicPath:
    points: np.ndarray  # [n, dim]
    length: float
    converged: bool
    end_error: float
    best_step: int

    @property
    def n_steps(self) -> int:
        return int(self.points.shape[0])


# =============================================================================
# MÉTRICA RIEMANNIANA (ATRATOR) - VETORIZADA
# =============================================================================

class SimpleRiemannianMetric:
    """
    Métrica com "atratores" que encarecem direções tangenciais, favorecendo cortar
    em direção ao atrator (comportamento de "canalizar" a geodésica).

    g(x) = I + Σ_k w_k(x) * P_tang_k
    P_tang_k = I - n_k n_k^T
    w_k(x) = strength / (1 + alpha * ||x - a_k||^2)
    """

    def __init__(self, dim: int, attractors: np.ndarray, cfg: MetricConfig):
        self.dim = int(dim)
        self.attractors = np.asarray(attractors, dtype=np.float64)  # [K, dim]
        self.cfg = cfg

    def metric_at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        g = np.eye(self.dim, dtype=np.float64)

        if self.attractors.size == 0:
            return g

        diffs = self.attractors - x[None, :]               # [K, dim]
        dist_sq = np.einsum("kd,kd->k", diffs, diffs)      # [K]
        mask = dist_sq > 1e-12
        if not np.any(mask):
            return g

        diffs = diffs[mask]
        dist_sq = dist_sq[mask]
        inv_dist = 1.0 / np.sqrt(dist_sq)
        n = diffs * inv_dist[:, None]                      # [K, dim]

        w = self.cfg.attractor_strength / (1.0 + self.cfg.alpha * dist_sq)  # [K]

        # g = I + Σ w(I - nn^T) = (1 + Σw)I - Σ w(nn^T)
        sum_w = float(np.sum(w))
        g *= (1.0 + sum_w)

        # Σ w(nn^T) = n^T (w * n)
        wn = n * w[:, None]                                # [K, dim]
        g -= n.T @ wn                                      # [dim, dim]

        # Garantia leve de SPD (evita singularidade numérica em casos extremos)
        # (se quiser, pode comentar)
        g += 1e-9 * np.eye(self.dim, dtype=np.float64)

        return g

    def christoffel_at_active(self, x: np.ndarray, active_dims: int) -> np.ndarray:
        """
        Retorna Γ^k_ij apenas no bloco active_dims (shape [ad, ad, ad]).
        Diferenças finitas centralizadas.
        """
        ad = int(min(active_dims, self.dim))
        eps = self.cfg.eps_fd

        g = self.metric_at(x)
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            return np.zeros((ad, ad, ad), dtype=np.float64)

        # dg[p, i, j] = ∂_p g_{ij}
        dg = np.zeros((ad, self.dim, self.dim), dtype=np.float64)
        for p in range(ad):
            x_plus = np.array(x, copy=True)
            x_minus = np.array(x, copy=True)
            x_plus[p] += eps
            x_minus[p] -= eps
            g_plus = self.metric_at(x_plus)
            g_minus = self.metric_at(x_minus)
            dg[p] = (g_plus - g_minus) / (2.0 * eps)

        gamma = np.zeros((ad, ad, ad), dtype=np.float64)

        # Γ^k_ij = 1/2 g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        # Só computamos índices até ad; l também até ad por custo.
        for k in range(ad):
            for i in range(ad):
                for j in range(ad):
                    # soma em l
                    term = 0.0
                    for l in range(ad):
                        tijl = dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        term += g_inv[k, l] * tijl
                    gamma[k, i, j] = 0.5 * term

        return gamma


# =============================================================================
# GEODESIC FLOW
# =============================================================================

class SimpleGeodesicFlow:
    def __init__(self, dim: int, metric: SimpleRiemannianMetric, cfg: FlowConfig):
        self.dim = int(dim)
        self.metric = metric
        self.cfg = cfg

    def _speed2(self, x: np.ndarray, v: np.ndarray) -> float:
        g = self.metric.metric_at(x)
        return float(v @ (g @ v))

    def _integrate_ivp(
        self,
        start: np.ndarray,
        v0: np.ndarray,
        end: np.ndarray | None = None,
        steps: int | None = None,
    ) -> SimpleGeodesicPath:
        """
        Integra equação geodésica com semi-implicit Euler + renormalização opcional.
        Se end for fornecido, aplica critérios de parada (patience/best-cut).
        """
        dt = self.cfg.dt
        max_steps = int(steps if steps is not None else self.cfg.max_steps)
        ad = int(min(self.cfg.active_dims, self.dim))

        x = np.array(start, dtype=np.float64, copy=True)
        v = np.array(v0, dtype=np.float64, copy=True)

        points = [x.copy()]
        total_len = 0.0

        # energia alvo (para reduzir drift): s2 = v^T g v
        s2_target = self._speed2(x, v) if self.cfg.energy_renorm else None

        best_step = 0
        best_err = float("inf")
        worse_count = 0

        for step in range(max_steps):
            gamma = self.metric.christoffel_at_active(x, ad)  # [ad,ad,ad]

            # a_k = - Γ^k_ij v^i v^j
            v_ad = v[:ad]
            a_ad = -np.einsum("kij,i,j->k", gamma, v_ad, v_ad)  # [ad]
            a = np.zeros_like(v)
            a[:ad] = a_ad

            # Semi-implicit Euler: v <- v + a dt ; x <- x + v dt
            v = v + a * dt

            if self.cfg.vel_damping > 0.0:
                v *= (1.0 - self.cfg.vel_damping)

            # Renormaliza energia para manter v^T g v ~ constante
            if self.cfg.energy_renorm and s2_target is not None:
                s2 = self._speed2(x, v)
                if s2 > self.cfg.speed_floor:
                    v *= np.sqrt(max(s2_target, 1e-12) / (s2 + 1e-12))

            x = x + v * dt
            points.append(x.copy())

            # comprimento incremental
            g_here = self.metric.metric_at(x)
            ds = np.sqrt(max(float(v @ (g_here @ v)), 0.0)) * dt
            total_len += ds

            # critério de parada com alvo
            if end is not None:
                err = float(np.linalg.norm(x - end))
                if err < best_err:
                    best_err = err
                    best_step = step + 1
                    worse_count = 0
                else:
                    worse_count += 1

                if err < self.cfg.tol:
                    # convergiu de verdade ao destino
                    pts = np.array(points, dtype=np.float64)
                    return SimpleGeodesicPath(
                        points=pts,
                        length=total_len,
                        converged=True,
                        end_error=err,
                        best_step=best_step,
                    )

                if worse_count >= self.cfg.patience:
                    # para de piorar: interrompe
                    break

        # se não convergiu: corta no melhor passo para não reportar um passeio infinito
        pts = np.array(points, dtype=np.float64)
        if end is not None and best_step > 0:
            pts = pts[: best_step + 1]

        end_err = float(np.linalg.norm(pts[-1] - end)) if end is not None else float("nan")
        # recomputa comprimento do path cortado (mais honesto)
        length_cut = 0.0
        for i in range(1, len(pts)):
            vi = (pts[i] - pts[i - 1]) / dt
            gi = self.metric.metric_at(pts[i])
            length_cut += np.sqrt(max(float(vi @ (gi @ vi)), 0.0)) * dt

        return SimpleGeodesicPath(
            points=pts,
            length=length_cut,
            converged=False,
            end_error=end_err,
            best_step=best_step,
        )

    def shortest_path(self, start: np.ndarray, end: np.ndarray) -> SimpleGeodesicPath:
        """
        Aproxima BVP start->end via shooting:
        ajusta v0 para minimizar ||x(T; v0) - end||.

        Observação: é "shooting light" (sem jacobiano). Bom o suficiente para demo.
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)

        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-12:
            return SimpleGeodesicPath(points=np.array([start]), length=0.0, converged=True, end_error=0.0, best_step=0)

        # define um horizonte T e passos coerentes com dt
        # (T maior => mais chance do IVP alcançar o alvo)
        dt = self.cfg.dt
        T = max(1.0, dist * 3.0)  # heurística simples: deixa "tempo" suficiente
        steps = int(min(max(T / dt, 30), self.cfg.max_steps))

        # chute inicial
        v0 = delta / max(T, 1e-12)

        best_path = None
        best_err = float("inf")

        for it in range(self.cfg.shooting_iters):
            path = self._integrate_ivp(start, v0, end=end, steps=steps)

            if path.end_error < best_err:
                best_err = path.end_error
                best_path = path

            if path.converged:
                return path

            # ajuste simples na direção do erro final + line-search
            err_vec = path.points[-1] - end
            base_err = path.end_error

            lr = self.cfg.lr
            accepted = False
            for _ in range(6):  # 6 tentativas de line search
                v_try = v0 - (lr / max(T, 1e-12)) * err_vec
                path_try = self._integrate_ivp(start, v_try, end=end, steps=steps)

                if path_try.end_error < base_err:
                    v0 = v_try
                    accepted = True
                    if path_try.converged: return path_try
                    break
                lr *= 0.5

            if not accepted:
                v0 = v0 - (0.1 * self.cfg.lr / max(T, 1e-12)) * err_vec

        # retorna o melhor encontrado (cortado no best_step pela integração)
        return best_path if best_path is not None else self._integrate_ivp(start, v0, end=end, steps=steps)


# =============================================================================
# DEMO
# =============================================================================

def make_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / (n + 1e-12)

def make_attractors_near_segment(a: np.ndarray, b: np.ndarray, k: int, offset: float, rng: np.random.Generator) -> np.ndarray:
    """
    Gera atratores próximos ao segmento a->b, com deslocamento perpendicular.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = make_unit(b - a)

    attrs = []
    for i in range(k):
        t = (i + 1) / (k + 1)
        base = (1.0 - t) * a + t * b

        p = rng.normal(size=a.shape)
        p = p - float(p @ d) * d  # perpendicular a d
        p = make_unit(p)

        attrs.append(base + offset * p)

    return np.asarray(attrs, dtype=np.float64)

def compute_euclid_polyline(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.outer(1 - t, a) + np.outer(t, b)

def metric_polyline_length(points: np.ndarray, metric: SimpleRiemannianMetric) -> float:
    """Calcula comprimento Riemanniano de uma polilinha."""
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i] - points[i-1]
        mid = 0.5 * (points[i] + points[i-1])
        g = metric.metric_at(mid)
        # ds² = dxᵀ g dx
        val = float(dx @ (g @ dx))
        total += np.sqrt(max(val, 0.0))
    return total

def demo():
    cfg = DemoConfig()
    mcfg = MetricConfig()
    fcfg = FlowConfig()

    print("=" * 60, flush=True)
    print("ALEXANDRIA :: DEMONSTRAÇÃO GEODÉSICA PURA (BENCHMARK ONLY)", flush=True)
    print("=" * 60, flush=True)

    rng = np.random.default_rng(cfg.seed)

    print(f"1) Configuração: dim={cfg.dim}, attractors={cfg.num_attractors}", flush=True)
    
    # PULANDO A DEMO DE PATH ÚNICO PARA IR DIRETO AO BENCHMARK

    print("2) Gerando pontos dummy A e B para setup (unit sphere)...", flush=True)
    A = make_unit(rng.normal(size=cfg.dim))
    B = make_unit(rng.normal(size=cfg.dim))

    print("3) Criando atratores próximos...", flush=True)
    attractors = make_attractors_near_segment(A, B, cfg.num_attractors, offset=0.30, rng=rng)

    print("4) Inicializando métrica e motor geodésico...", flush=True)
    metric = SimpleRiemannianMetric(cfg.dim, attractors, mcfg)
    flow = SimpleGeodesicFlow(cfg.dim, metric, fcfg)

    # 8) Benchmark em Batch
    print(flush=True)
    print(">>> BENCHMARK START (10 iterations)...", flush=True)
    bench_stats = run_benchmark(flow, cfg, mcfg, attractors)
    print(f"   🏆 Convergência: {bench_stats['convergence_rate']:.1%}", flush=True)
    print(f"   🎯 Erro Mediano: {bench_stats['median_error']:.2e}", flush=True)
    print(f"   📐 Ratio Mediano: {bench_stats['median_ratio']:.4f}", flush=True)
    print(f"   ⏱️  Tempo Médio:  {bench_stats['avg_time']:.4f}s", flush=True)

    print(flush=True)
    print("=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)

    return {
        "benchmark": bench_stats
    }

def check_energy_drift(path: SimpleGeodesicPath, metric: SimpleRiemannianMetric) -> tuple[np.ndarray, dict]:
    """Calcula vT g v ao longo do caminho para verificar conservação de energia."""
    # Reconstrói velocidades aproximadas (ou poderia ter salvo no integrador)
    points = path.points
    if len(points) < 2:
        return np.array([1.0]), {"min": 1.0, "max": 1.0, "mean": 1.0, "drift_rel": 0.0}
        
    energies = []
    # v ~ (x[i+1] - x[i]) / dt ?? Não temos dt aqui fácil sem passar config.
    # Mas a conservação independe do dt se medirmos direcionalmente.
    # Melhor: Se o integrador normaliza, vT g v deve ser constante.
    
    # Vamos assumir que points estão espaçados por dt (aproximadamente)
    # Mas para verificar "vT g v", precisamos do v real. 
    # Como não salvamos v no path, vamos estimar via diferenças finitas locais.
    
    for i in range(len(points) - 1):
        # Direção tangencial
        dx = points[i+1] - points[i]
        # v_est = dx / ||dx||_g  -> isso daria energia 1 sempre por definição se normalizarmos.
        # O teste é ver se o passo 'dx' mantém tamanho constante na métrica ao longo do tempo?
        # Sim, em geodesica parametrizada por comprimento de arco (ou tempo cte), ||v||_g deve ser cte.
        
        mid = (points[i] + points[i+1]) / 2
        g = metric.metric_at(mid)
        energy_sq = np.dot(dx, np.dot(g, dx)) # proporcional a energy se dt fixo
        energies.append(energy_sq)
        
    energies = np.array(energies)
    stats = {
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "mean": float(np.mean(energies)),
        "drift_rel": float((np.max(energies) - np.min(energies)) / (np.mean(energies) + 1e-12))
    }
    return energies, stats

def run_benchmark(flow: SimpleGeodesicFlow, dcfg: DemoConfig, mcfg: MetricConfig, attractors: np.ndarray) -> dict:
    """Roda N pares aleatórios e coleta estatísticas."""
    import time
    
    n_samples = 10
    rng = np.random.default_rng(dcfg.seed + 1)
    
    converged_count = 0
    errors = []
    ratios = []
    times = []
    
    for _ in range(n_samples):
        A = make_unit(rng.normal(size=dcfg.dim))
        B = make_unit(rng.normal(size=dcfg.dim))
        
        t0 = time.time()
        path = flow.shortest_path(A, B)
        dt = time.time() - t0
        times.append(dt)
        
        errors.append(path.end_error)
        
        if path.converged:
            converged_count += 1
            
        # Calcula ratio se convergiu (ou mesmo se não, pra ver o quão ruim foi)
        # Ratio = Geodesic / Baseline (Euclid approx in Metric)
        # Para ser rápido, usamos uma polilinha grosseira pro baseline
        base_pts = compute_euclid_polyline(A, B, path.n_steps)
        base_len = metric_polyline_length(base_pts, flow.metric)
        geo_len = path.length
        
        r_val = geo_len / (base_len + 1e-12)
        ratios.append(r_val)
    
    # Escreve arquivo "sinal" com resultados no nome para leitura via ls
    # Evita problemas de captura de output
    fname = f"RESULT_Conv_{converged_count / n_samples:.2f}_Err_{float(np.median(errors)):.4f}_Ratio_{float(np.median(ratios)):.4f}.res"
    with open(fname, "w") as f:
        f.write("Done")
        
    return {
        "convergence_rate": converged_count / n_samples,
        "median_error": float(np.median(errors)),
        "median_ratio": float(np.median(ratios)),
        "avg_time": float(np.mean(times))
    }


if __name__ == "__main__":
    demo()
