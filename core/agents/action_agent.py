"""
DEPRECATED: Este módulo foi refatorado para core/agents/action/

Este arquivo mantém compatibilidade reversa exportando tudo da nova estrutura.
Recomenda-se atualizar seus imports para:

    from core.agents.action import ActionAgent, ActionType, ...

ao invés de:

    from core.agents.action_agent import ActionAgent, ActionType, ...
"""

import warnings

# Emitir warning de depreciação
warnings.warn(
    "Importing from 'core.agents.action_agent' is deprecated. "
    "Use 'core.agents.action' instead. "
    "The entire module has been refactored into a modular structure.",
    DeprecationWarning,
    stacklevel=2
)

# Re-exportar tudo da nova estrutura para manter compatibilidade
from core.agents.action import *  # noqa: F401, F403
from core.agents.action import __all__  # noqa: F401

# Manter função de serialização para compatibilidade
def _serialize_action_result(action_result):
    """Serializa ActionResult com conversão de enums para string"""
    if action_result is None:
        return {}
    
    return {
        "action_id": action_result.action_id,
        "action_type": action_result.action_type.value if hasattr(action_result.action_type, 'value') else str(action_result.action_type),
        "status": action_result.status.value if hasattr(action_result.status, 'value') else str(action_result.status),
        "start_time": action_result.start_time.isoformat() if action_result.start_time else None,
        "end_time": action_result.end_time.isoformat() if action_result.end_time else None,
        "result_data": action_result.result_data,
        "error_message": action_result.error_message,
        "evidence_generated": action_result.evidence_generated,
        "evidence_type": action_result.evidence_type.value if action_result.evidence_type and hasattr(action_result.evidence_type, 'value') else str(action_result.evidence_type) if action_result.evidence_type else None,
        "duration": action_result.duration
    }
