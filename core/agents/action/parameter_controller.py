"""
Alexandria - Parameter Controller
Parameter management for the Action Agent system.

This module handles system parameter adjustments, validation, and history tracking.
"""

import os
import logging
from typing import Dict, List, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ParameterController:
    """Controller para ajustes de parâmetros do sistema"""
    
    def __init__(self):
        self.parameter_history = []
        self.supported_parameters = {
            # Parâmetros do V11 Vision Encoder
            "V11_BETA": {"min": 0.1, "max": 10.0, "current": 1.0},
            "V11_LEARNING_RATE": {"min": 0.0001, "max": 0.1, "current": 0.001},
            "V11_BATCH_SIZE": {"min": 16, "max": 128, "current": 32},
            "V11_EPOCHS": {"min": 10, "max": 200, "current": 50},
            
            # Parâmetros do SFS
            "SFS_CHUNK_SIZE": {"min": 256, "max": 2048, "current": 512},
            "SFS_THRESHOLD": {"min": 0.1, "max": 1.0, "current": 0.5},
            
            # Parâmetros do Causal Engine
            "CAUSAL_VARIANCE_THRESHOLD": {"min": 0.01, "max": 1.0, "current": 0.1},
            "CAUSAL_MIN_EDGE_WEIGHT": {"min": 0.01, "max": 0.5, "current": 0.05},
        }
    
    def adjust_parameter(self, param_name: str, new_value: Union[float, int, str]) -> bool:
        """Ajusta parâmetro do sistema"""
        try:
            if param_name not in self.supported_parameters:
                logger.error(f"Parâmetro não suportado: {param_name}")
                return False
            
            param_config = self.supported_parameters[param_name]
            
            # Validar tipo e range
            if isinstance(new_value, (int, float)):
                if "min" in param_config and new_value < param_config["min"]:
                    return False
                if "max" in param_config and new_value > param_config["max"]:
                    return False
            
            # Registrar no histórico
            self.parameter_history.append({
                "timestamp": datetime.now().isoformat(),
                "parameter": param_name,
                "old_value": param_config.get("current"),
                "new_value": new_value,
                "change_type": "adjustment"
            })
            
            # Atualizar valor atual
            param_config["current"] = new_value
            
            # Aplicar no ambiente se aplicável
            os.environ[param_name] = str(new_value)
            
            logger.info(f"Parâmetro ajustado: {param_name} = {new_value}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao ajustar parâmetro {param_name}: {e}")
            return False
    
    def reset_parameter(self, param_name: str) -> bool:
        """Reseta parâmetro para valor padrão"""
        if param_name in self.supported_parameters:
            default_value = self.supported_parameters[param_name].get("default")
            if default_value is not None:
                return self.adjust_parameter(param_name, default_value)
        return False
    
    def get_parameter(self, param_name: str) -> Any:
        """Obtém valor atual do parâmetro"""
        return self.supported_parameters.get(param_name, {}).get("current")
    
    def get_parameter_history(self, param_name: str = None) -> List[Dict[str, Any]]:
        """Retorna histórico de alterações de parâmetro"""
        if param_name:
            return [entry for entry in self.parameter_history if entry["parameter"] == param_name]
        return self.parameter_history
