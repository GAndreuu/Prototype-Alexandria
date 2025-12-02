import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.utils.state_manager import get_mycelial

def render_sidebar():
    """Renders the common sidebar for all pages."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.title("Alexandria")
        st.markdown("*Cognitive System v2.0*")
        
        st.markdown("---")
        
        # System Status
        st.subheader("Status")
        st.success("🟢 Core: ONLINE")
        st.info("🟡 API: BYPASSED (Local)")
        
        # Mycelial Stats
        mycelial = get_mycelial()
        stats = mycelial.get_network_stats()
        
        st.markdown("---")
        st.markdown("**🧠 Brain Stats**")
        st.metric("Observações", stats['total_observations'])
        st.metric("Conexões", stats['active_connections'])
        st.metric("Densidade", f"{stats['density']:.4%}")
        
        st.markdown("---")
        st.markdown("### Navegação")
        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/1_🧠_Dashboard.py", label="📊 Dashboard", icon="📊")
        st.page_link("pages/2_🍄_Mycelial_Brain.py", label="🍄 Mycelial Brain", icon="🍄")
        st.page_link("pages/3_🕸️_Knowledge_Graph.py", label="🕸️ Knowledge Graph", icon="🕸️")
        st.page_link("pages/4_🔮_Abduction.py", label="🔮 Abduction", icon="🔮")
        st.page_link("pages/5_💥_Collider.py", label="💥 Collider", icon="💥")
