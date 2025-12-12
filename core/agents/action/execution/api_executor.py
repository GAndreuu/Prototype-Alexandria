"""
Alexandria - API Executor
Handles API call execution with security validation.
"""

import logging
import requests
from typing import Dict, Any
from datetime import datetime

from ..types import ActionResult, ActionStatus, ActionType
from ..security_controller import SecurityController

logger = logging.getLogger(__name__)


def execute_api_call(
    parameters: Dict[str, Any],
    security_controller: SecurityController,
    action_id: str
) -> ActionResult:
    """
    Executa chamada de API com validação de segurança.
    
    Args:
        parameters: Parâmetros da API call (url, method, headers, data, timeout)
        security_controller: Instância do SecurityController
        action_id: ID da ação
        
    Returns:
        ActionResult com resultado da execução
    """
    url = parameters.get("url")
    method = parameters.get("method", "GET").upper()
    headers = parameters.get("headers", {})
    data = parameters.get("data")
    timeout = parameters.get("timeout", 30)
    
    # Validação de segurança
    if not security_controller.validate_api_call(url):
        raise ValueError(f"API não permitida: {url}")
    
    # Executar requisição
    logger.info(f"Executando API call: {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Método HTTP não suportado: {method}")
        
        result_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "response_size": len(response.content),
            "success": 200 <= response.status_code < 300
        }
        
        # Tentar parsear JSON se possível
        try:
            result_data["json_response"] = response.json()
        except:
            result_data["text_response"] = response.text[:1000]  # Primeiros 1000 chars
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.API_CALL,
            status=ActionStatus.COMPLETED,
            start_time=datetime.now(),
            result_data=result_data
        )
        
    except Exception as e:
        raise Exception(f"Erro na API call: {str(e)}")
