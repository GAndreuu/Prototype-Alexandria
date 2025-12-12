import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.components.sidebar import render_sidebar

st.set_page_config(page_title="Knowledge Graph | Alexandria", page_icon="🕸️", layout="wide")
render_sidebar()

st.title("🕸️ Knowledge Graph")
st.markdown("Visualização interativa da topologia do conhecimento.")

col_viz, col_ctrl = st.columns([3, 1])

with col_viz:
    html_path = Path("network_viz_3d.html")
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("Grafo não gerado. Use o botão ao lado para criar.")

with col_ctrl:
    st.subheader("Controles")
    
    if st.button("🕸️ Gerar Grafo 3D"):
        with st.spinner("Calculando conexões semânticas..."):
            # Executa o script de visualização
            os.system("python scripts/visualize_knowledge_graph.py")
            st.rerun()
            
    st.markdown("---")
    st.markdown("""
    **Legenda:**
    - **Nós**: Chunks de conhecimento
    - **Arestas**: Similaridade > 0.7
    - **Cores**: Domínios (Physics, Bio, AI, etc.)
    """)
    
    st.info("Use o mouse para rotacionar, dar zoom e inspecionar nós.")
