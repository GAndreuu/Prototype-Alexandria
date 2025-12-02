import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.utils.state_manager import init_session_state
from interface.components.sidebar import render_sidebar

# --- Config ---
st.set_page_config(
    page_title="Alexandria Control Deck",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Style ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .stButton>button { background-color: #21262d; color: #58a6ff; border: 1px solid #30363d; }
    .stButton>button:hover { background-color: #30363d; }
    h1, h2, h3 { color: #58a6ff; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 5px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# --- Init ---
init_session_state()
render_sidebar()

# --- Main Content ---
st.title("🧬 Alexandria Control Deck")
st.markdown("### Bem-vindo ao Sistema Cognitivo")

st.info("👈 Selecione um módulo na barra lateral para começar.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📊 Dashboard
    Monitoramento em tempo real de CPU, RAM e status do banco de dados LanceDB.
    """)
    if st.button("Ir para Dashboard"):
        st.switch_page("pages/1_🧠_Dashboard.py")

    st.markdown("""
    #### 🍄 Mycelial Brain
    Interface de raciocínio em tempo real. Converse com o sistema e veja a rede neural reagir.
    """)
    if st.button("Ir para Cérebro"):
        st.switch_page("pages/2_🍄_Mycelial_Brain.py")

with col2:
    st.markdown("""
    #### 🕸️ Knowledge Graph
    Visualização 3D interativa de todo o conhecimento indexado.
    """)
    if st.button("Explorar Grafo"):
        st.switch_page("pages/3_🕸️_Knowledge_Graph.py")

    st.markdown("""
    #### 🔮 Abduction Engine
    Motor de geração de hipóteses e descoberta de lacunas.
    """)
    if st.button("Gerar Hipóteses"):
        st.switch_page("pages/4_🔮_Abduction.py")
