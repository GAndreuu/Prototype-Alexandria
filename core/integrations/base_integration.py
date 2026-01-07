"""
Base Integration Classes
========================

Classes base abstratas para todas as integrações Bridge ↔ Compositional.
Reduz boilerplate e padroniza comportamento.

Uso:
    from core.integrations.base_integration import BaseCompositionalIntegration
    
    class MyIntegration(BaseCompositionalIntegration):
        def _default_config(self):
            return MyConfig()

Autor: Alexandria Project
Versão: 1.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO BASE
# =============================================================================

@dataclass
class BaseIntegrationConfig:
    """Configuração base para todas as integrações."""
    
    # Comum a todas
    use_geodesic: bool = True
    use_curvature: bool = True
    use_field_energy: bool = True
    
    # Limites de performance
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    timeout_seconds: float = 30.0
    
    # Debug
    debug: bool = False
    log_level: str = "INFO"


# =============================================================================
# MÉTRICAS BASE
# =============================================================================

@dataclass
class IntegrationMetrics:
    """Métricas comuns de execução."""
    
    calls: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_error: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def record_call(self, duration_ms: float, error: Optional[str] = None):
        """Registra uma chamada."""
        self.calls += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.calls
        
        if error:
            self.errors += 1
            self.last_error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "calls": self.calls,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.calls),
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "last_error": self.last_error,
            **self.custom
        }


# =============================================================================
# INTEGRAÇÃO BASE ABSTRATA
# =============================================================================

class BaseCompositionalIntegration(ABC):
    """
    Classe base abstrata para todas as integrações Bridge ↔ Compositional.
    
    Fornece:
    - Inicialização padronizada
    - Métricas automáticas
    - Validação de componentes
    - Helpers comuns
    
    Subclasses devem implementar:
    - _default_config(): Retorna configuração padrão
    - _validate_specific(): Validações específicas (opcional)
    """
    
    def __init__(
        self,
        bridge,
        compositional=None,
        config=None
    ):
        """
        Inicializa a integração.
        
        Args:
            bridge: VQVAEManifoldBridge obrigatório
            compositional: CompositionalReasoner opcional
            config: Configuração específica ou None para defaults
        """
        self.bridge = bridge
        self.compositional = compositional
        self.config = config or self._default_config()
        
        # Métricas
        self._metrics = IntegrationMetrics()
        
        # Validar
        self._validate_base()
        self._validate_specific()
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def _default_config(self):
        """Retorna configuração padrão. Deve ser implementado."""
        pass
    
    def _validate_base(self):
        """Validações base comuns."""
        if self.bridge is None:
            raise ValueError(f"{self.__class__.__name__} requires a bridge")
    
    def _validate_specific(self):
        """Validações específicas. Override se necessário."""
        pass
    
    # =========================================================================
    # HELPERS COMUNS
    # =========================================================================
    
    def _embed(self, vector: np.ndarray) -> Any:
        """Embede vetor no manifold via bridge."""
        if hasattr(self.bridge, 'embed'):
            return self.bridge.embed(vector)
        return vector
    
    def _geodesic(self, start: np.ndarray, end: np.ndarray) -> Optional[np.ndarray]:
        """Calcula geodésica entre dois pontos."""
        if self.compositional is not None and hasattr(self.compositional, 'reason'):
            result = self.compositional.reason(start, end)
            return getattr(result, 'points', None)
        
        # Fallback: interpolação linear
        n_steps = 10
        return np.linspace(start, end, n_steps)
    
    def _geodesic_length(self, start: np.ndarray, end: np.ndarray) -> float:
        """Calcula comprimento geodésico."""
        path = self._geodesic(start, end)
        if path is None:
            return np.linalg.norm(end - start)
        
        # Soma dos segmentos
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def _curvature_at(self, point: np.ndarray) -> float:
        """Curvatura local no ponto."""
        if self.bridge is not None and hasattr(self.bridge, 'metric'):
            try:
                g = self.bridge.metric.metric_at(point)
                # Curvatura escalar aproximada
                return float(np.trace(g) - len(point))
            except:
                pass
        return 0.0
    
    def _field_energy_at(self, point: np.ndarray) -> float:
        """Energia livre do campo no ponto."""
        if self.bridge is not None and hasattr(self.bridge, 'field'):
            try:
                return self.bridge.field.free_energy_at(point)
            except:
                pass
        return 0.0
    
    # =========================================================================
    # DECORATORS PARA MÉTRICAS
    # =========================================================================
    
    def _timed_call(self, func, *args, **kwargs):
        """Executa função com timing automático."""
        start = time.time()
        error = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            self._metrics.record_call(duration_ms, error)
        
        return result
    
    # =========================================================================
    # API PÚBLICA
    # =========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da integração."""
        return {
            "integration": self.__class__.__name__,
            "has_bridge": self.bridge is not None,
            "has_compositional": self.compositional is not None,
            "config_type": self.config.__class__.__name__,
            "metrics": self._metrics.to_dict()
        }
    
    def reset_metrics(self):
        """Reseta métricas."""
        self._metrics = IntegrationMetrics()
    
    def health_check(self) -> Dict[str, bool]:
        """Verifica saúde dos componentes."""
        return {
            "bridge": self.bridge is not None,
            "compositional": self.compositional is not None,
            "config": self.config is not None,
            "bridge_connected": hasattr(self.bridge, 'vqvae') and self.bridge.vqvae is not None if self.bridge else False
        }


# =============================================================================
# FACTORY HELPER
# =============================================================================

def create_integration(
    integration_class,
    bridge,
    compositional=None,
    **config_kwargs
):
    """
    Factory helper para criar integrações.
    
    Args:
        integration_class: Classe da integração
        bridge: VQVAEManifoldBridge
        compositional: CompositionalReasoner opcional
        **config_kwargs: Parâmetros de configuração
    
    Returns:
        Instância da integração
    """
    # Obtém classe de config da integração
    config_class = None
    if hasattr(integration_class, '_default_config'):
        # Cria instância temporária para pegar config class
        temp = object.__new__(integration_class)
        temp.bridge = bridge
        temp.compositional = compositional
        config_class = type(temp._default_config())
    
    config = None
    if config_class and config_kwargs:
        config = config_class(**config_kwargs)
    
    return integration_class(bridge, compositional, config)


# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Base Integration - Teste")
    print("=" * 60)
    
    # Teste com implementação simples
    class TestConfig(BaseIntegrationConfig):
        test_param: float = 1.0
    
    class TestIntegration(BaseCompositionalIntegration):
        def _default_config(self):
            return TestConfig()
        
        def do_something(self, x: np.ndarray):
            return self._timed_call(lambda: x * 2)
    
    # Mock bridge
    class MockBridge:
        pass
    
    # Criar
    integration = TestIntegration(MockBridge())
    
    print(f"Health: {integration.health_check()}")
    print(f"Stats: {integration.stats()}")
    
    # Testar chamada com métricas
    result = integration.do_something(np.array([1, 2, 3]))
    print(f"Result: {result}")
    print(f"Stats after call: {integration.stats()}")
    
    print("\n✅ Base Integration funcionando!")
