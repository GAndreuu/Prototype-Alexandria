import streamlit as st
import requests
import json
import os
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuração da Página ---
st.set_page_config(
    page_title="Prototype Alexandria",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilo Cyberpunk/Futurista ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #21262d; 
        color: #58a6ff;
        border: 1px solid #30363d;
        border-radius: 6px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }
    .stTextInput>div>div>input {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
        text-align: center;
        font-weight: bold;
    }
    .status-ok { background-color: #0f5323; color: #e6ffec; border-color: #1a7f37; }
    .status-warn { background-color: #5a3e02; color: #fff8c5; border-color: #9a6700; }
    .status-err { background-color: #5a1e1e; color: #ffc5c5; border-color: #9a2e2e; }
    
    .metric-card {
        background-color: #161b22;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #30363d;
        margin-bottom: 5px;
    }
    .metric-label { font-size: 0.8em; color: #8b949e; }
    .metric-value { font-size: 1.2em; color: #c9d1d9; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Constantes ---
API_URL = "http://localhost:8000"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_FILE = Path("data/chat_history.json")

# --- Funções Auxiliares ---
def check_api():
    try:
        requests.get(f"{API_URL}/docs", timeout=1)
        return True
    except:
        return False

def get_status():
    try:
        return requests.get(f"{API_URL}/causal/status", timeout=1).json()
    except:
        return None

def get_system_stats():
    try:
        return requests.get(f"{API_URL}/system/stats", timeout=1).json()
    except:
        return {}

def get_learning_status():
    try:
        return requests.get(f"{API_URL}/learning/status", timeout=1).json()
    except:
        return {}

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar chat: {e}")

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Prototype Alexandria")
    st.markdown("### *Neural Interface*")
    
    api_online = check_api()
    if api_online:
        st.success("SYSTEM: ONLINE")
        status = get_status()
        stats = get_system_stats()
        
        # Dashboard
        if stats:
            st.markdown("### 🖥️ Hardware Monitor")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CPU</div>
                    <div class="metric-value">{stats.get('cpu_percent', 0)}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RAM</div>
                    <div class="metric-value">{stats.get('ram_percent', 0)}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Arquivos Indexados</div>
                <div class="metric-value">{stats.get('files_indexed', 0)} docs</div>
            </div>
            """, unsafe_allow_html=True)

            # V2 Status
            v2_status = get_learning_status()
            if v2_status and v2_status.get("status") == "active":
                 st.markdown(f"""
                <div class="metric-card" style="border-color: #8b949e;">
                    <div class="metric-label">Self-Feeding Cycle</div>
                    <div class="metric-value" style="color: #7ee787;">ACTIVE</div>
                    <div class="metric-label" style="font-size: 0.7em;">{v2_status.get('device', 'cpu')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Self-Feeding Cycle</div>
                    <div class="metric-value" style="color: #ff7b72;">INACTIVE</div>
                </div>
                """, unsafe_allow_html=True)
            
    else:
        st.error("SYSTEM: OFFLINE")
        status = None
        stats = {}

    st.markdown("---")
    st.markdown("### Configurações")
    mode = st.selectbox(
        "Modo de Raciocínio",
        ["hybrid", "local", "gemini"],
        index=0,
        help="Hybrid: TinyLlama + Gemini. Local: Offline. Gemini: Nuvem."
    )
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()
    
    st.markdown("---")
    st.info("v12.0.0-FINAL")

# --- Tabs Principais ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat & Oracle", 
    "📂 Memória & Ingestão", 
    "🧠 Cérebro Visual",
    "🔮 Cognição Avançada",
    "🌌 Deep Analysis"
])

# === TAB 1: CHAT ===
with tab1:
    st.header("💬 Interface de Comunicação")
    
    chat_mode = st.radio("Selecione o Canal:", ["📚 Chat com Livros (RAG)", "🤖 Chat Direto (LLM Puro)"], horizontal=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Exibir histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Digite sua mensagem..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_history(st.session_state.messages)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ *Processando pensamento...*")
            
            try:
                # Define se usa RAG ou não baseado na escolha
                use_rag = True if "Livros" in chat_mode else False
                
                payload = {"text": prompt, "mode": mode, "use_rag": use_rag}
                res = requests.post(f"{API_URL}/query", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    
                    # Mostrar evidências em expander APENAS se for modo Livros
                    if use_rag and data.get("evidence"):
                        with st.expander("🔍 Evidências Recuperadas (SFS)"):
                            for i, ev in enumerate(data["evidence"]):
                                st.markdown(f"**Evidência {i+1}:** {ev}")
                    elif use_rag and not data.get("evidence"):
                         st.warning("⚠️ Nenhuma evidência encontrada nos livros.")
                    
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    save_chat_history(st.session_state.messages)
                else:
                    placeholder.error(f"Erro na API: {res.text}")
            except Exception as e:
                placeholder.error(f"Erro de conexão: {e}")

# === TAB 2: MEMÓRIA & INGESTÃO ===
with tab2:
    st.header("📂 Gestão de Conhecimento (SFS)")
    
    tab2_1, tab2_2 = st.tabs(["📥 Ingestão", "🗑️ Gerenciador de Memória"])
    
    with tab2_1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Manual")
            uploaded_files = st.file_uploader(
                "Arraste arquivos aqui (.txt, .pdf, .md, .png, .jpg)", 
                type=['txt', 'pdf', 'md', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button(f"📥 Ingerir {len(uploaded_files)} Arquivos"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    success_count = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Salvar arquivo
                        file_path = UPLOAD_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        status_text.text(f"Processando: {uploaded_file.name}...")
                        
                        # Chamar API
                        try:
                            payload = {"file_path": str(file_path.absolute()), "type": "GEN"}
                            res = requests.post(f"{API_URL}/ingest", json=payload)
                            if res.status_code == 200:
                                success_count += 1
                            else:
                                st.error(f"Falha em {uploaded_file.name}: {res.text}")
                        except Exception as e:
                            st.error(f"Erro em {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"Concluído! {success_count}/{len(uploaded_files)} arquivos indexados.")
                    status_text.text("Pronto.")
                    time.sleep(2)
                    st.rerun()

        with col2:
            st.subheader("Pasta Mágica (Auto-Ingestão)")
            st.info("📁 Caminho: `data/magic_folder`")
            st.markdown("""
            **Como usar:**
            1. Abra a pasta `data/magic_folder` no seu Windows.
            2. Jogue arquivos lá dentro.
            3. O sistema detecta e ingere automaticamente!
            
            *(Execute `python scripts/auto_ingest.py` para ativar o monitoramento)*
            """)
            
            st.subheader("Status da Memória")
            if status:
                st.json({
                    "Index File": status.get("index_file_exists"),
                    "Ready for Analysis": status.get("ready_for_causal_analysis"),
                    "Files Indexed": stats.get("files_indexed", 0)
                })
            else:
                st.warning("Conecte a API para ver status.")

    with tab2_2:
        st.subheader("Gerenciador de Arquivos Indexados")
        
        # Obter lista de arquivos (precisamos de um endpoint para isso, ou ler do stats se tiver detalhe)
        # Como não temos endpoint de lista completa, vamos ler o arquivo de indice diretamente se possível
        # Ou melhor, vamos adicionar um endpoint simples de listagem no backend depois.
        # Por enquanto, vamos simular lendo o arquivo local se existir, já que estamos local.
        
        INDEX_FILE = Path("data/knowledge.sfs")
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                
                if not index_data:
                    st.info("Memória vazia.")
                else:
                    # Converter para DataFrame para tabela bonita
                    files_list = []
                    for fid, meta in index_data.items():
                        files_list.append({
                            "ID": fid,
                            "Arquivo": Path(meta['file_path']).name,
                            "Tipo": meta['modalidade'],
                            "Chunks": meta['chunks_count']
                        })
                    
                    st.dataframe(files_list, use_container_width=True)
                    
                    # Área de Exclusão
                    st.markdown("---")
                    st.subheader("🗑️ Excluir Arquivo")
                    file_to_delete = st.selectbox("Selecione para remover:", [f["ID"] for f in files_list], format_func=lambda x: f"{x} - {index_data[x]['file_path']}")
                    
                    if st.button("Confirmar Exclusão"):
                        try:
                            res = requests.delete(f"{API_URL}/memory/file?file_id={file_to_delete}")
                            if res.status_code == 200:
                                st.success("Arquivo removido da memória!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Erro ao deletar: {res.text}")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            except Exception as e:
                st.error(f"Erro ao ler índice local: {e}")
        else:
            st.warning("Índice de memória não encontrado.")

# === TAB 3: CÉREBRO VISUAL ===
with tab3:
    st.header("🧠 Topologia Neural & Causalidade")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Controles")
        
        # Status
        if status and status.get("topology_trained"):
            st.markdown('<div class="status-box status-ok">✅ Topologia Treinada</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-warn">⚠️ Não Treinada</div>', unsafe_allow_html=True)
            if st.button("🚀 Treinar Topologia"):
                with st.spinner("Treinando..."):
                    try:
                        train_path = "data/training/corpus.txt"
                        if not os.path.exists(train_path):
                            files = list(UPLOAD_DIR.glob("*.txt"))
                            if files: train_path = str(files[0])
                        
                        requests.post(f"{API_URL}/train_topology?path={train_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

        if status and status.get("causal_graph_built"):
            st.markdown('<div class="status-box status-ok">✅ Grafo Construído</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-warn">⚠️ Não Construído</div>', unsafe_allow_html=True)
            if st.button("🕸️ Construir Grafo"):
                with st.spinner("Analisando..."):
                    requests.post(f"{API_URL}/causal/build_graph")
                    st.rerun()
                    
        if st.button("🔄 Atualizar Visualização"):
            st.rerun()

    with c2:
        st.subheader("Visualização do Manifold (Top 20 Clusters)")
        try:
            res = requests.get(f"{API_URL}/causal/graphviz")
            if res.status_code == 200:
                dot_data = res.json().get("dot")
                if dot_data:
                    st.graphviz_chart(dot_data)
                else:
                    st.info("Sem dados para visualizar.")
            else:
                st.warning("Não foi possível carregar o grafo.")
        except:
            st.error("Erro ao conectar com API de visualização.")

# === TAB 4: COGNIÇÃO AVANÇADA ===
with tab4:
    st.header("🔮 Ferramentas Cognitivas Avançadas")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Explicação Causal")
        causal_query = st.text_input("Pergunte 'Por que...' ou 'Qual a causa de...'")
        if st.button("Explicar Causalidade"):
            with st.spinner("Raciocinando..."):
                try:
                    payload = {"text": causal_query, "explain": True}
                    res = requests.post(f"{API_URL}/causal/explain", json=payload)
                    if res.status_code == 200:
                        st.info(res.json())
                    else:
                        st.error(f"Erro: {res.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")

    with col_b:
        st.subheader("Gerador de Hipóteses (Abduction)")
        if st.button("🧪 Gerar Novas Hipóteses"):
            with st.spinner("Abduzindo conhecimento..."):
                try:
                    payload = {"max_hypotheses": 5}
                    res = requests.post(f"{API_URL}/abduction/generate_hypotheses", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"{data['count']} Hipóteses Geradas")
                        for hyp in data['hypotheses']:
                            with st.expander(f"Hipótese: {hyp['hypothesis_text']}"):
                                st.json(hyp)
                    else:
                        st.error(f"Erro: {res.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.markdown("---")
    st.subheader("💤 Ciclo de Sono (Consolidação de Memória)")
    if st.button("🌙 Forçar Sono & Aprendizado (Self-Feeding)"):
        with st.spinner("Sonhando e consolidando memórias no V2..."):
            try:
                # Simular vetores de "sonho" (memórias recentes aleatórias)
                import numpy as np
                dream_vectors = np.random.normal(0, 0.1, (10, 384)).tolist()
                
                res = requests.post(f"{API_URL}/learning/trigger", json=dream_vectors)
                if res.status_code == 200:
                    data = res.json()
                    metrics = data.get("result_data", {}).get("learning_metrics", {})
                    st.success("Ciclo de sono concluído! Memórias consolidadas.")
                    st.json(metrics)
                else:
                    st.error(f"Erro no sono: {res.text}")
            except Exception as e:
                st.error(f"Erro: {e}")

# === TAB 5: DEEP ANALYSIS ===
with tab5:
    st.header("🌌 Deep Analysis Dashboard (Real-Time)")
    
    # 1. Manifold Visualization
    st.subheader("Visualização do Manifold (3D/2D)")
    
    col_viz_1, col_viz_2 = st.columns([3, 1])
    
    with col_viz_2:
        viz_mode = st.radio("Modo de Visualização", ["3D Space", "2D Projection"], index=0)
        color_by = st.selectbox("Colorir por", ["modality", "cluster", "source"], index=0)
        limit_points = st.slider("Limite de Pontos", 100, 5000, 2000)
        
        if st.button("🔄 Atualizar Dados"):
            st.rerun()
            
    with col_viz_1:
        try:
            with st.spinner("Carregando dados do manifold..."):
                res = requests.get(f"{API_URL}/visualization/manifold_data?limit={limit_points}")
                if res.status_code == 200:
                    data = res.json()
                    points = data.get("points", [])
                    
                    if points:
                        df = pd.DataFrame(points)
                        
                        if viz_mode == "3D Space":
                            fig = px.scatter_3d(
                                df, x='x', y='y', z='z',
                                color=color_by,
                                hover_data=['content', 'id', 'source'],
                                title=f"Manifold Neural ({len(points)} pontos)",
                                template="plotly_dark",
                                opacity=0.7
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.scatter(
                                df, x='x', y='y',
                                color=color_by,
                                hover_data=['content', 'id', 'source'],
                                title=f"Projeção 2D ({len(points)} pontos)",
                                template="plotly_dark",
                                opacity=0.7
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Nenhum dado no manifold para visualizar. Ingira arquivos primeiro.")
                else:
                    st.error("Erro ao carregar dados do manifold.")
        except Exception as e:
            st.error(f"Erro na visualização: {e}")

    st.markdown("---")
    
    # 2. Evolutionary Stats
    st.subheader("📈 Estatísticas Evolutivas (Self-Feeding)")
    
    try:
        res = requests.get(f"{API_URL}/visualization/evolution_stats")
        if res.status_code == 200:
            stats_data = res.json()
            history = stats_data.get("history", [])
            
            if history:
                # Converter para DataFrame
                history_df = pd.DataFrame([
                    {
                        "timestamp": h["timestamp"],
                        "total_loss": h["metrics"]["total_loss"],
                        "recon_loss": h["metrics"]["recon_loss"],
                        "vq_loss": h["metrics"]["vq_loss"]
                    }
                    for h in history
                ])
                
                # Gráfico de Loss
                fig_loss = px.line(
                    history_df, x="timestamp", y=["total_loss", "recon_loss", "vq_loss"],
                    title="Evolução da Perda (Loss) Neural",
                    template="plotly_dark",
                    labels={"value": "Loss", "variable": "Métrica"}
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Métricas Recentes
                latest = history[-1]["metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Loss", f"{latest['total_loss']:.4f}")
                c2.metric("Reconstruction", f"{latest['recon_loss']:.4f}")
                c3.metric("VQ Commitment", f"{latest['vq_loss']:.4f}")
                c4.metric("Orthogonal", f"{latest['ortho_loss']:.4f}")
                
            else:
                st.info("Sem histórico de evolução ainda. O sistema precisa 'aprender' primeiro.")
        else:
            st.error("Erro ao carregar estatísticas.")
            
    except Exception as e:
        st.error(f"Erro nas estatísticas: {e}")
