import streamlit as st
import sys
from pathlib import Path

# Add project root to path and scripts
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.components.sidebar import render_sidebar
from scripts import collide

st.set_page_config(page_title="Semantic Collider | Alexandria", page_icon="💥", layout="wide")
render_sidebar()

st.title("💥 Semantic Collider")
st.markdown("Encontre conexões ocultas entre domínios diferentes.")

col1, col2 = st.columns(2)
with col1:
    source = st.text_input("Fonte (Contexto A)", value="Jogo do Exterminador")
with col2:
    target = st.text_input("Alvo (Contexto B)", value="papers")
    
if st.button("🚀 Iniciar Colisão"):
    with st.spinner("Colidindo vetores e gerando hipóteses..."):
        try:
            collide.collide(source, target)
            st.success("Colisão concluída!")
            
            # Read Report
            report_path = Path("collision_report.txt")
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()
                st.markdown("### 📄 Relatório de Colisão")
                st.code(report_content, language="markdown")
            else:
                st.error("Relatório não encontrado.")
        except Exception as e:
            st.error(f"Erro na colisão: {e}")
