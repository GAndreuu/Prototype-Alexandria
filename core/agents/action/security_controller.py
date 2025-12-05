"""
Alexandria - Security Controller
Security and audit control for Action Agent operations.

This module handles API validation, rate limiting, and audit logging.
"""

import os
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

from .types import ActionType, ActionResult

logger = logging.getLogger(__name__)


class SecurityController:
    """Controller de segurança para execução de ações"""
    
    def __init__(self):
        # Whitelist de APIs permitidas
        self.allowed_apis = self._load_allowed_apis()
        self.rate_limits = {}
        self.audit_log = []
        self.blocked_domains = set()
        
    def _load_allowed_apis(self) -> List[str]:
        """Carrega lista de APIs permitidas do ambiente"""
        apis_env = os.environ.get("ALLOWED_APIS", "")
        return [api.strip() for api in apis_env.split(",") if api.strip()]
    
    def validate_api_call(self, url: str) -> bool:
        """Valida se a URL da API é permitida"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Verificar domínio bloqueado
            if any(blocked in domain for blocked in self.blocked_domains):
                return False
            
            # Verificar whitelist
            if self.allowed_apis:
                return any(allowed in domain for allowed in self.allowed_apis)
            
            # Se não há whitelist, negar por padrão
            return False
            
        except Exception as e:
            logger.error(f"Erro na validação de API: {e}")
            return False
    
    def check_rate_limit(self, action_type: ActionType, user_id: str = "system") -> bool:
        """Verifica rate limiting"""
        key = f"{action_type.value}_{user_id}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Limpar timestamps antigos (últimos 5 minutos)
        self.rate_limits[key] = [
            ts for ts in self.rate_limits[key] 
            if now - ts < 300  # 5 minutos
        ]
        
        # Verificar limite (10 ações por 5 minutos)
        if len(self.rate_limits[key]) >= 10:
            return False
        
        # Adicionar timestamp atual
        self.rate_limits[key].append(now)
        return True
    
    def log_action(self, action: ActionResult, details: Dict[str, Any]):
        """Registra ação no log de auditoria"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "status": action.status.value,
            "details": details
        }
        
        self.audit_log.append(log_entry)
        
        # Manter apenas últimas 1000 entradas
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna log de auditoria"""
        return self.audit_log[-limit:] if self.audit_log else []
