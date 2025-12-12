#!/bin/bash

# Alexandria Unified Runner
# =========================
# Lança todo o sistema localmente:
# 1. API Backend (Port 8000)
# 2. Interface Neural (Port 8501)
# 3. System Runner V2 (Processamento Cognitivo em Background)

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}      ALEXANDRIA COGNITIVE SYSTEM V12          ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Verificar ambiente Python
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Ambiente virtual .venv encontrado${NC}"
    source .venv/bin/activate
else
    echo -e "${BLUE}ℹ Usando ambiente python do sistema (certifique-se das dependências)${NC}"
fi

# Criar diretórios de log
mkdir -p logs

# 1. Iniciar Backend API
echo -e "\n${GREEN}[1/3] Iniciando API Backend (Uvicorn)...${NC}"
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   PID: $BACKEND_PID | Log: logs/backend.log"

# 2. Iniciar Interface
echo -e "\n${GREEN}[2/3] Iniciando Neural Interface (Streamlit)...${NC}"
nohup streamlit run interface/app.py --server.port=8501 --server.address=0.0.0.0 > logs/interface.log 2>&1 &
UI_PID=$!
echo "   PID: $UI_PID | Log: logs/interface.log"

# 3. Iniciar Cognitive Runner (Opcional)
echo -e "\n${GREEN}[3/3] Iniciando Ciclo Cognitivo (System Runner V2)...${NC}"
read -p "   Deseja iniciar o processamento em background? (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    nohup python3 scripts/system_runner_v2.py --cycles 1000 --workers 4 > logs/runner.log 2>&1 &
    RUNNER_PID=$!
    echo "   PID: $RUNNER_PID | Log: logs/runner.log"
else
    RUNNER_PID=""
fi

echo -e "\n${BLUE}===============================================${NC}"
echo -e "Sistema Operacional!"
echo -e "   API Docs: http://localhost:8000/docs"
echo -e "   Interface: http://localhost:8501"
echo -e "${BLUE}===============================================${NC}"
echo "Pressione [CTRL+C] para encerrar todos os processos..."

# Trap para matar processos filhos ao sair
trap "kill $BACKEND_PID $UI_PID $RUNNER_PID 2>/dev/null; echo ' Sistema encerrado.'; exit" INT TERM

# Manter script rodando
wait
