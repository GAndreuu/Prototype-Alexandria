"""
Geodesic Flow
=============

Motor de integração de geodésicas (caminhos de menor energia) em variedades Riemannianas.
Implementa métodos robustos para resolver o Boundary Value Problem (BVP) entre dois pontos.

Algoritmos:
- Integração Semi-Implícita (Euler Symplectic-ish)
- Shooting Method com Line Search
- Renormalização de Energia (vT g v)

Autor: Alexandria System
Versão: 2.0 (Ported from Demo)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from core.field.metric import RiemannianMetric
import logging

logger = logging.getLogger(__name__)

# mHC Safety Layer import
try:
    from core.field.manifold_constraints import cap_kinetic_energy
    _HAS_MHC = True
except ImportError:
    _HAS_MHC = False
    def cap_kinetic_energy(v, current_e, max_e, metric=None):
        """Fallback simples."""
        if current_e > max_e and current_e > 1e-12:
            return v * np.sqrt(max_e / current_e)
        return v

@dataclass
class GeodesicConfig:
    """Configuração do fluxo geodésico."""
    dt: float = 0.02
    max_steps: int = 200
    active_dims: int = 24       # Dimensões ativas para cálculo de curvatura
    energy_renorm: bool = True  # Mantém vT g v constante (conservação de energia)
    vel_damping: float = 0.0    # Amortecimento (para relaxamento)
    speed_floor: float = 1e-6
    use_scipy_integrator: bool = False  # Usar scipy.integrate.solve_ivp ao invés de Euler

    # Parâmetros do Shooting Method (BVP)
    shooting_iters: int = 35
    lr: float = 0.35            # Taxa de aprendizado inicial para correção de velocidade
    tol: float = 1e-2           # Tolerância de convergência (distância final)
    patience: int = 35          # Passos sem melhora antes de abortar integração
    
    # mHC Safety Layer: Limites de energia para evitar divergência
    # Ref: mHC paper (DeepSeek-AI, 2025) - não-expansividade
    use_mhc_energy_cap: bool = True
    max_energy_ratio: float = 10.0  # Máximo de 10x a energia inicial


@dataclass
class GeodesicPath:
    """Representa um caminho geodésico calculado."""
    points: np.ndarray          # [n_steps, dim]
    length: float               # Comprimento métrico acumulado
    converged: bool             # Se atingiu o alvo dentro da tolerância
    end_error: float            # Distância final ao alvo
    best_step: int              # Passo onde a distância foi mínima (para cortes)

    @property
    def n_steps(self) -> int:
        return int(self.points.shape[0])


class GeodesicFlow:
    """
    Gerencia o cálculo de geodésicas na variedade.
    """

    def __init__(self, manifold: 'DynamicManifold', metric: RiemannianMetric, config: Optional[GeodesicConfig] = None):
        self.manifold = manifold
        self.metric = metric
        self.config = config or GeodesicConfig()
        self.dim = getattr(manifold, 'base_dim', 64) # Fallback se não definido

    def _speed2(self, x: np.ndarray, v: np.ndarray) -> float:
        """Calcula energia cinética vT g(x) v."""
        g = self.metric.metric_at(x)
        return float(v @ (g @ v))

    def _integrate_ivp(
        self,
        start: np.ndarray,
        v0: np.ndarray,
        end: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
    ) -> GeodesicPath:
        """
        Integra equação geodésica como Problema de Valor Inicial (IVP).
        
        Args:
            start: Ponto inicial
            v0: Velocidade inicial
            end: Alvo opcional (para early stopping e métricas)
            steps: Override de max_steps
            
        Returns:
            GeodesicPath integrado
        """
        dt = self.config.dt
        max_steps = int(steps if steps is not None else self.config.max_steps)
        ad = int(min(self.config.active_dims, self.dim))

        x = np.array(start, dtype=np.float64, copy=True)
        v = np.array(v0, dtype=np.float64, copy=True)

        points = [x.copy()]
        total_len = 0.0

        # Alvo de energia para renormalização
        s2_target = self._speed2(x, v) if self.config.energy_renorm else None

        best_step = 0
        best_err = float("inf")
        worse_count = 0

        for step in range(max_steps):
            # Calcula Christoffel symbols apenas nas dimensões ativas
            gamma = self.metric.christoffel_at_active(x, ad)  # [ad, ad, ad]

            # Aceleração geodésica: a^k = - Γ^k_ij v^i v^j
            v_ad = v[:ad]
            a_ad = -np.einsum("kij,i,j->k", gamma, v_ad, v_ad)
            
            a = np.zeros_like(v)
            a[:ad] = a_ad

            # Semi-implicit Euler
            v = v + a * dt

            if self.config.vel_damping > 0.0:
                v *= (1.0 - self.config.vel_damping)

            # Renormalização de energia
            if self.config.energy_renorm and s2_target is not None:
                s2 = self._speed2(x, v)
                if s2 > self.config.speed_floor:
                    # Scaling factor: sqrt(target / current)
                    v *= np.sqrt(max(s2_target, 1e-12) / (s2 + 1e-12))
            
            # =================================================================
            # mHC SAFETY LAYER: Energy Capping
            # Ref: mHC paper - não-expansividade via ||H||_2 ≤ 1
            # Limita energia cinética para evitar explosão em regiões de alta
            # curvatura onde Christoffel pode amplificar a velocidade
            # =================================================================
            if self.config.use_mhc_energy_cap:
                current_energy = self._speed2(x, v)
                max_allowed = s2_target * self.config.max_energy_ratio if s2_target else 10.0
                
                if current_energy > max_allowed:
                    v = cap_kinetic_energy(v, current_energy, max_allowed)
                    logger.debug(f"Geodesic energy capped at step {step}")

            x = x + v * dt
            points.append(x.copy())

            # Comprimento incremental (Riemanniano)
            g_here = self.metric.metric_at(x)
            ds = np.sqrt(max(float(v @ (g_here @ v)), 0.0)) * dt
            total_len += ds

            # Verificação de alvo (Critérios de parada)
            if end is not None:
                err = float(np.linalg.norm(x - end))
                
                # Rastreamento do melhor ponto
                if err < best_err:
                    best_err = err
                    best_step = step + 1
                    worse_count = 0
                else:
                    worse_count += 1

                # Convergência
                if err < self.config.tol:
                    return GeodesicPath(
                        points=np.array(points, dtype=np.float64),
                        length=total_len,
                        converged=True,
                        end_error=err,
                        best_step=best_step,
                    )

                # Early stopping por não-melhora (overshoot ou preso)
                if worse_count >= self.config.patience:
                    break

        # Se saiu do loop sem convergir:
        # Corta o caminho no ponto de maior proximidade (Best Cut)
        pts = np.array(points, dtype=np.float64)
        if end is not None and best_step > 0 and best_step < len(pts):
            pts = pts[: best_step + 1]
            
            # Recalcula comprimento do caminho cortado
            # (Poderia ter acumulado até best_step, mas recalcular é mais seguro aqui)
            length_cut = 0.0
            # Aproximação rápida linear para comprimento cortado se necessário,
            # ou usar o acumulado proporcional. Aqui, apenas usamos total_len se full,
            # ou recalculamos. Recalcular métrica é caro.
            # Vamos estimar proporcionalmente para evitar custo extra:
            length_cut = total_len * (best_step / len(points)) 
        else:
            length_cut = total_len

        end_err = float(np.linalg.norm(pts[-1] - end)) if end is not None else float("nan")

        return GeodesicPath(
            points=pts,
            length=length_cut,
            converged=False,
            end_error=end_err,
            best_step=best_step,
        )

    def shortest_path(self, start: np.ndarray, end: np.ndarray, max_iterations: int = 35) -> GeodesicPath:
        """
        Calcula o caminho mais curto (geodésica) entre start e end (BVP).
        Usa Shooting Method com ajuste iterativo velocidade inicial.
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)

        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-12:
            return GeodesicPath(np.array([start]), 0.0, True, 0.0, 0)

        # Heurística para tempo de integração
        dt = self.config.dt
        T = max(1.0, dist * 3.0) 
        steps = int(min(max(T / dt, 30), self.config.max_steps))

        # Chute inicial (velocidade Euclidiana)
        v0 = delta / max(T, 1e-12)

        best_path = None
        best_err = float("inf")
        
        # Override config iterations se fornecido
        iterations = max_iterations or self.config.shooting_iters

        for it in range(iterations):
            path = self._integrate_ivp(start, v0, end=end, steps=steps)

            if path.end_error < best_err:
                best_err = path.end_error
                best_path = path

            if path.converged:
                return path

            # Correção de velocidade (Shooting Update)
            err_vec = path.points[-1] - end
            base_err = path.end_error

            # Line Search para garantir melhora
            lr = self.config.lr
            accepted = False
            for _ in range(6):
                # Gradiente descendente aproximado: v_new = v_old - lr * erro
                v_try = v0 - (lr / max(T, 1e-12)) * err_vec
                path_try = self._integrate_ivp(start, v_try, end=end, steps=steps)

                if path_try.end_error < base_err:
                    v0 = v_try
                    accepted = True
                    if path_try.converged: 
                        return path_try
                    break
                    
                lr *= 0.5 # Reduz passo se piorou

            # Se line search falhou em achar melhora, tenta perturbação menor ou continua
            if not accepted:
                 v0 = v0 - (0.1 * self.config.lr / max(T, 1e-12)) * err_vec

        # Retorna o melhor encontrado
        return best_path if best_path is not None else self._integrate_ivp(start, v0, end=end, steps=steps)

    def propagate_activation(self, start_point: 'ManifoldPoint', steps: int = 3, decay: float = 0.5) -> List['ManifoldPoint']:
        """
        Simula propagação de ativação (como uma 'onda') usando geodésicas radiais.
        (Ainda placeholder/básico, ideal seria wave-equation solver)
        """
        # TODO: Implementar propagação real baseada em frentes de onda
        return [start_point]
