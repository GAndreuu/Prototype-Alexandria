# ASI Monolith V12 - Production Dockerfile
FROM python:3.10-slim

# Configuração de Ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements_real.txt .

# Instalar dependências Python
# Remover versões específicas de torch se necessário para usar a versão CPU/CUDA correta do container
RUN pip install --no-cache-dir -r requirements_real.txt

# Copiar código fonte
COPY . .

# Criar diretórios de dados
RUN mkdir -p data/uploads data/training data/indexed data/logs models/fine_tuned

# Expor portas
# 8000: FastAPI Backend
# 8501: Streamlit UI
EXPOSE 8000
EXPOSE 8501

# Script de entrada
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
