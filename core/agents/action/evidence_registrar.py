"""
Alexandria - Evidence Registrar
Registers test evidence in the SFS multi-modal system.

This module transforms action results and simulations into structured evidence
that can be queried by the system.
"""

import json
import logging
import time
from typing import Dict, Any
from datetime import datetime

from .types import ActionResult, ActionStatus, EvidenceType

logger = logging.getLogger(__name__)


class EvidenceRegistrar:
    """
    EvidenceRegistrar - Registra evidências de testes no SFS multi-modal.
    
    Responsabilidade: Transformar resultados de ações e simulações em
    evidências estruturadas que podem ser consultadas pelo sistema.
    """
    
    def __init__(self, action_agent, sfs_instance):
        self.action_agent = action_agent
        self.sfs = sfs_instance
        self.evidence_index = {}
        
    def register_action_evidence(self, action_result: ActionResult) -> str:
        """Registra evidência de ação no SFS"""
        evidence_id = f"ACTION_EVID_{action_result.action_id}"
        
        evidence_content = f"""
# Evidência de Ação Executada

**ID da Ação**: {action_result.action_id}
**Tipo**: {action_result.action_type.value}
**Status**: {action_result.status.value}
**Duração**: {action_result.duration:.2f}s

## Dados do Resultado
```json
{json.dumps(action_result.result_data, indent=2)}
```

## Contexto
- Data/Hora: {action_result.start_time}
- Tipo de Evidência: {'Suporte' if action_result.evidence_type == EvidenceType.SUPPORTING else 'Refutação' if action_result.evidence_type == EvidenceType.CONTRADICTING else 'Neutro'}
"""
        
        # Registrar no SFS
        try:
            # Salvar arquivo de evidência
            evidence_file = self.action_agent.sfs_path / f"{evidence_id}.md"
            with open(evidence_file, 'w', encoding='utf-8') as f:
                f.write(evidence_content)
            
            # Indexar no SFS multi-modal
            chunks_count = self.sfs.index_file(str(evidence_file))
            
            self.evidence_index[evidence_id] = {
                "evidence_type": "action_result",
                "file_path": str(evidence_file),
                "chunks_indexed": chunks_count,
                "registration_time": datetime.now().isoformat(),
                "evidence_strength": "high" if action_result.status == ActionStatus.COMPLETED else "low"
            }
            
            logger.info(f"Evidência de ação registrada: {evidence_id}")
            return evidence_id
            
        except Exception as e:
            logger.error(f"Erro ao registrar evidência de ação: {e}")
            raise
    
    def register_simulation_evidence(self, simulation_data: Dict[str, Any]) -> str:
        """Registra evidência de simulação no SFS"""
        evidence_id = f"SIM_EVID_{int(time.time())}"
        
        evidence_content = f"""
# Evidência de Simulação Executada

**Simulação**: {simulation_data.get('simulation_name', 'unknown')}
**Concluída em**: {simulation_data.get('completed_at', '')}

## Parâmetros Testados
```json
{json.dumps(simulation_data, indent=2)}
```

## Análise de Resultados
Esta evidência foi gerada através de simulação para testar hipóteses causais do sistema ASI.
"""
        
        try:
            # Salvar arquivo de evidência
            evidence_file = self.action_agent.sfs_path / f"{evidence_id}.md"
            with open(evidence_file, 'w', encoding='utf-8') as f:
                f.write(evidence_content)
            
            # Indexar no SFS
            chunks_count = self.sfs.index_file(str(evidence_file))
            
            self.evidence_index[evidence_id] = {
                "evidence_type": "simulation_result",
                "file_path": str(evidence_file),
                "chunks_indexed": chunks_count,
                "registration_time": datetime.now().isoformat(),
                "evidence_strength": "high",
                "simulation_type": simulation_data.get('simulation_name', 'unknown')
            }
            
            logger.info(f"Evidência de simulação registrada: {evidence_id}")
            return evidence_id
            
        except Exception as e:
            logger.error(f"Erro ao registrar evidência de simulação: {e}")
            raise
    
    def get_evidence_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas das evidências registradas"""
        total_evidence = len(self.evidence_index)
        
        if total_evidence == 0:
            return {"total_evidence": 0}
        
        # Contar por tipo
        evidence_types = {}
        for evidence in self.evidence_index.values():
            ev_type = evidence.get("evidence_type", "unknown")
            if ev_type not in evidence_types:
                evidence_types[ev_type] = 0
            evidence_types[ev_type] += 1
        
        return {
            "total_evidence": total_evidence,
            "evidence_types": evidence_types,
            "total_chunks": sum(e.get("chunks_indexed", 0) for e in self.evidence_index.values())
        }
