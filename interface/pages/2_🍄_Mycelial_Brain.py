import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.components.sidebar import render_sidebar
from interface.utils.state_manager import get_mycelial, get_learner, get_encoder

st.set_page_config(page_title="Mycelial Brain | Alexandria", page_icon="🍄", layout="wide")
render_sidebar()

st.title("🍄 Mycelial Brain Interface")
st.markdown("Interaja com o sistema e veja a ativação neural em tempo real.")

col_input, col_viz = st.columns([1, 2])

with col_input:
    st.subheader("Stimulus Input")
    input_text = st.text_area("Digite um pensamento ou conceito:", height=150, 
                             value="A entropia aumenta com o tempo, mas a vida organiza a matéria.")
    
    if st.button("🧠 Processar (Encode -> Observe -> Reason)"):
        with st.spinner("Processando no Córtex Neural..."):
            learner = get_learner()
            encoder = get_encoder()
            mycelial = get_mycelial()
            
            # 1. Encode Text -> Vector
            embedding = encoder.encode([input_text])[0]
            
            # 2. Encode Vector -> Indices (Monolith)
            with torch.no_grad():
                t_emb = torch.tensor([embedding], dtype=torch.float32).to(learner.device)
                out = learner.model(t_emb)
                indices = out['indices'].cpu().numpy().flatten()
                
            st.session_state['last_indices'] = indices
            st.success(f"Encoded Indices: {indices}")
            
            # 3. Observe (Learn)
            mycelial.observe(indices)
            st.toast("Padrão aprendido via Hebbian Learning!", icon="🎓")
            
            # 4. Reason (Propagate)
            new_indices, activation = mycelial.reason(indices, steps=5)
            st.session_state['last_activation'] = activation
            st.session_state['reasoned_indices'] = new_indices
            
            st.info(f"Reasoned Indices: {new_indices}")

with col_viz:
    st.subheader("Neural Activation Map")
    
    if 'last_activation' in st.session_state:
        activation = st.session_state['last_activation']
        
        # Visualizar ativação por Head
        # Activation shape: (4, 256)
        
        df_list = []
        for h in range(4):
            for i in range(256):
                val = activation[h, i]
                if val > 0.01: # Filter noise
                    df_list.append({'Head': f"H{h}", 'Index': i, 'Activation': val})
        
        if df_list:
            df = pd.DataFrame(df_list)
            fig = px.bar(df, x='Index', y='Activation', color='Head', 
                         title="Ativação Neural por Cabeça (Multi-Head)",
                         template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nenhuma ativação significativa detectada.")
            
    else:
        st.info("Aguardando input para visualizar ativação...")

st.markdown("---")
st.subheader("Conceitos Associados (Decodificados)")

if 'reasoned_indices' in st.session_state:
    # Tentar decodificar o que esses indices significam (se tivéssemos o decoder de texto)
    # Como não temos text decoder, vamos mostrar os vizinhos semânticos no LanceDB
    # Isso é um "hack" para mostrar o que o cérebro está "pensando"
    
    st.markdown("**Interpretação Semântica (via LanceDB):**")
    # TODO: Implementar busca reversa no LanceDB pelos clusters ativados
    st.caption("Funcionalidade de decodificação textual em desenvolvimento (Requer VQ-VAE Decoder treinado em texto).")
