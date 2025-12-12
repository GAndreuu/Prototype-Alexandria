import sys
import os
from loguru import logger
from config import settings

# Criar diretório de logs se não existir
log_dir = os.path.join(settings.DATA_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# Configuração do Logger
def setup_logger():
    # Remover handlers padrão
    logger.remove()
    
    # Handler para Console (Colorido e Conciso)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Handler para Arquivo (JSON Estruturado para análise futura)
    logger.add(
        os.path.join(log_dir, "system.log"),
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        level="DEBUG",
        serialize=True  # JSON format
    )
    
    # Handler para Arquivo Legível (Texto)
    logger.add(
        os.path.join(log_dir, "system_readable.log"),
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.info("Logger estruturado inicializado com sucesso.")

# Inicializar logger ao importar
setup_logger()
