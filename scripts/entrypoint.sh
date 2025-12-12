#!/bin/bash

# Iniciar Backend em background
echo "🚀 Iniciando Prototype Alexandria Backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Aguardar backend iniciar
sleep 5

# Iniciar Frontend
echo "🎨 Iniciando Neural Interface..."
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
