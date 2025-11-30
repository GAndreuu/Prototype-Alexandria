import streamlit as st
import sys
import os
from pathlib import Path
import time
import pandas as pd
from PIL import Image

# Add root to path to allow imports
sys.path.append(str(Path(__file__).parent))

# Import core modules directly
try:
    from scripts import collide
    from core.reasoning.abduction_engine import AbductionEngine
    from core.agents.action_agent import ActionAgent
except ImportError:
    # Handle case where scripts is not a package or path issues
    sys.path.append(str(Path(__file__).parent / "scripts"))
    import collide
    from core.reasoning.abduction_engine import AbductionEngine
    from core.agents.action_agent import ActionAgent

from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig
from core.reasoning.neural_learner import V2Learner
from sentence_transformers import SentenceTransformer
from config import settings

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
    .report-box { background-color: #161b22; padding: 15px; border-radius: 5px; border: 1px solid #30363d; font-family: monospace; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Alexandria Control Deck")
st.markdown("*Interface de Abstração para o Sistema Cognitivo*")

# --- Sidebar ---
with st.sidebar:
    st.header("Status do Sistema")
    st.success("🟢 Core: ONLINE")
    st.info("🟡 API: BYPASSED (Local Mode)")
    
    st.markdown("---")
    st.markdown("**Módulos Ativos:**")
    st.checkbox("Semantic Collider", value=True, disabled=True)
    st.checkbox("Abduction Engine", value=True, disabled=True)
    st.checkbox("Neural Learner", value=True, disabled=True)
    st.checkbox("Mycelial Reasoning", value=True, disabled=True)

    # Initialize Mycelial Engine in Session State
    if 'mycelial' not in st.session_state:
        st.session_state.mycelial = MycelialReasoning()
    
    if 'learner' not in st.session_state:
        # Lazy load learner to avoid startup delay
        st.session_state.learner = None
    
    if 'encoder' not in st.session_state:
        st.session_state.encoder = None

def get_learner():
    if st.session_state.learner is None:
        with st.spinner("Carregando Neural Learner (Monolith V13)..."):
            st.session_state.learner = V2Learner()
    return st.session_state.learner

def get_encoder():
    if st.session_state.encoder is None:
        with st.spinner("Carregando Encoder Textual..."):
            st.session_state.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return st.session_state.encoder


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💥 Semantic Collider", "🕸️ Topologia (Static)", "🕸️ Rede (Interactive)", "🧠 Abdução & Hipóteses", "⚙️ Ações & Demo", "🍄 Raciocínio Micelial"])


# === TAB 1: COLLIDER ===
with tab1:
    st.header("Colisor Semântico")
    st.markdown("Encontre conexões ocultas entre domínios diferentes.")
    
    col1, col2 = st.columns(2)
    with col1:
        source = st.text_input("Fonte (Contexto A)", value="Jogo do Exterminador")
    with col2:
        target = st.text_input("Alvo (Contexto B)", value="papers")
        
    if st.button("🚀 Iniciar Colisão"):
        with st.spinner("Colidindo vetores e gerando hipóteses..."):
            try:
                # Redirect stdout to capture logs if needed, but we rely on file
                collide.collide(source, target)
                st.success("Colisão concluída!")
                
                # Read Report
                report_path = Path("collision_report.txt")
                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    st.markdown("### 📄 Relatório de Colisão")
                    st.markdown(f'<div class="report-box">{report_content}</div>', unsafe_allow_html=True)
                else:
                    st.error("Relatório não encontrado.")
            except Exception as e:
                st.error(f"Erro na colisão: {e}")

# === TAB 2: TOPOLOGY ===
with tab2:
    st.header("Topologia do Conhecimento")
    
    col_viz, col_ctrl = st.columns([3, 1])
    
    with col_viz:
        img_path = Path("topology_viz_3d.png")
        if img_path.exists():
            image = Image.open(img_path)
            st.image(image, caption="Visualização 3D do Espaço Semântico", use_container_width=True)
        else:
            st.warning("Nenhuma visualização gerada ainda.")
            
    with col_ctrl:
        st.markdown("### Controles")
        if st.button("🔄 Regenerar Gráfico"):
            with st.spinner("Amostrando vetores e plotando..."):
                os.system("python scripts/visualize_topology.py")
                st.rerun()
        
        st.info("O gráfico mostra a distância semântica entre Papers (Clusters) e Livros (Outliers).")

# === TAB 3: NETWORK ===
with tab3:
    st.header("Rede Semântica Interativa")
    st.markdown("Grafo 3D navegável. **Nós** = Chunks, **Arestas** = Similaridade > 0.7.")
    
    col_net, col_ctrl_net = st.columns([3, 1])
    
    with col_net:
        html_path = Path("network_viz_3d.html")
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=False)
        else:
            st.warning("Grafo de rede não gerado.")
            
    with col_ctrl_net:
        if st.button("🕸️ Gerar Rede 3D"):
            with st.spinner("Calculando conexões..."):
                os.system("python scripts/visualize_network.py")
                st.rerun()
        st.info("Use o mouse para girar, zoom e hover nos nós.")

# === TAB 4: ABDUCTION ===
with tab4:
    st.header("Motor de Abdução")
    st.markdown("Detecção automática de lacunas e geração de conhecimento.")
    
    if st.button("🔍 Escanear Lacunas & Gerar Hipóteses"):
        with st.spinner("O AbductionEngine está pensando..."):
            engine = AbductionEngine()
            gaps = engine.detect_knowledge_gaps()
            hypotheses = engine.generate_hypotheses()
            
            st.subheader(f"Lacunas Detectadas: {len(gaps)}")
            for gap in gaps:
                with st.expander(f"🧩 {gap.description}"):
                    st.write(f"**Tipo:** {gap.gap_type}")
                    st.write(f"**Prioridade:** {gap.priority_score:.2f}")
            
            st.subheader(f"Hipóteses Geradas: {len(hypotheses)}")
            for h in hypotheses:
                with st.expander(f"💡 {h.hypothesis_text}"):
                    st.write(f"**Confiança:** {h.confidence_score:.2f}")
                    st.write(f"**Status:** {h.validation_status}")
                    st.write(f"**Fonte:** {h.source_cluster} -> **Alvo:** {h.target_cluster}")

# === TAB 5: ACTIONS ===
with tab5:
    st.header("Agente de Ação")
    st.markdown("Execução de testes e validações no mundo real.")
    
    st.info("Este módulo permite que a IA execute simulações ou buscas para validar suas hipóteses.")
    
    if st.button("▶️ Rodar Demo de Capacidades (Full Cycle)"):
        with st.spinner("Executando ciclo completo: Abdução -> Ação -> Aprendizado..."):
            # Run the demo script and capture output
            import subprocess
            result = subprocess.run(["python", "scripts/demo_capabilities.py"], capture_output=True, text=True)
            
            st.code(result.stdout, language="bash")
            
            if result.stderr:
                st.error("Erros:")
                st.code(result.stderr)

# === TAB 6: MYCELIAL REASONING ===
with tab6:
    st.header("🍄 Raciocínio Micelial")
    st.markdown("Rede Hebbian de co-ativação entre códigos latentes.")
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        st.subheader("Simulação de Pensamento")
        input_text = st.text_area("Input de Texto (Pensamento):", value="A inteligência artificial evolui através de padrões recursivos.")
        
        if st.button("🧠 Processar (Encode -> Observe -> Reason)"):
            learner = get_learner()
            encoder = get_encoder()
            mycelial = st.session_state.mycelial
            
            # 1. Encode Text -> Vector
            embedding = encoder.encode([input_text])[0] # 384D
            
            # 2. Encode Vector -> Indices (Monolith)
            # Precisamos de um método no learner para pegar indices
            # O learner.encode retorna z_q, mas precisamos dos indices
            # Vamos usar o modelo direto
            import torch
            with torch.no_grad():
                t_emb = torch.tensor([embedding], dtype=torch.float32).to(learner.device)
                out = learner.model(t_emb)
                indices = out['indices'].cpu().numpy().flatten()
                
            st.info(f"Indices Originais: {indices}")
            
            # 3. Observe (Learn)
            mycelial.observe(indices)
            st.success("Padrão observado e conexões reforçadas!")
            
            # 4. Reason (Propagate)
            new_indices, activation = mycelial.reason(indices, steps=5)
            st.info(f"Indices Raciocinados: {new_indices}")
            
            # 5. Decode Reasoned
            # Hack: reconstruir z_q a partir dos indices novos
            # O learner não tem esse método exposto facilmente, vamos tentar via decoder direto se possível
            # Mas o decoder precisa de z_q, não indices.
            # O quantizer tem os embeddings.
            
            # Visualização da Ativação
            st.bar_chart(activation.T) # Transpose para ter heads como series ou algo assim
            
    with col_m2:
        st.subheader("Estatísticas da Rede")
        if st.button("Atualizar Stats"):
            pass # Rerun
            
        stats = st.session_state.mycelial.get_network_stats()
        st.metric("Observações Totais", stats['total_observations'])
        st.metric("Conexões Ativas", stats['active_connections'])
        st.metric("Densidade", f"{stats['density']:.4%}")
        
        st.markdown("### Top Conexões")
        top_conns = st.session_state.mycelial.get_strongest_connections(5)
        for c in top_conns:
            st.text(f"H{c['head']}: {c['from']} -> {c['to']} ({c['strength']:.2f})")
