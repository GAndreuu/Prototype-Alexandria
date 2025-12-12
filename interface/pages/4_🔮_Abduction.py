import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.components.sidebar import render_sidebar
from interface.utils.state_manager import get_abduction_engine

st.set_page_config(page_title="Abduction Engine | Alexandria", page_icon="🔮", layout="wide")
render_sidebar()

st.title("🔮 Abduction Engine")
st.markdown("Motor de geração de hipóteses e descoberta de lacunas.")

engine = get_abduction_engine()

tab1, tab2 = st.tabs(["🔍 Lacunas", "💡 Hipóteses"])

with tab1:
    st.subheader("Detecção de Lacunas")
    if st.button("Escanear Lacunas"):
        with st.spinner("Analisando grafo causal..."):
            gaps = engine.detect_knowledge_gaps()
            st.success(f"{len(gaps)} lacunas detectadas.")
            
            for gap in gaps:
                with st.expander(f"🧩 {gap.description} (Score: {gap.priority_score:.2f})"):
                    st.write(f"**ID:** {gap.gap_id}")
                    st.write(f"**Tipo:** {gap.gap_type}")
                    st.write(f"**Clusters Afetados:** {gap.affected_clusters}")

with tab2:
    st.subheader("Geração de Hipóteses")
    if st.button("Gerar Hipóteses"):
        with st.spinner("Abduzindo conhecimento..."):
            hypotheses = engine.generate_hypotheses()
            st.success(f"{len(hypotheses)} hipóteses geradas.")
            
            for h in hypotheses:
                with st.expander(f"💡 {h.hypothesis_text}"):
                    st.write(f"**Confiança:** {h.confidence_score:.2f}")
                    st.write(f"**Status:** {h.validation_status}")
                    st.write(f"**Fonte:** {h.source_cluster} -> **Alvo:** {h.target_cluster}")
                    
                    if st.button(f"Validar Hipótese {h.id}", key=h.id):
                        with st.spinner("Validando..."):
                            valid = engine.validate_hypothesis(h.id)
                            if valid:
                                st.success("✅ Validada!")
                            else:
                                st.error("❌ Rejeitada.")
                            st.rerun()
