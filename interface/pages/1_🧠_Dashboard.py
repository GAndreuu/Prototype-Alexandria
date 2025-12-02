import streamlit as st
import psutil
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from interface.components.sidebar import render_sidebar
from interface.utils.state_manager import get_mycelial
from core.memory.storage import LanceDBStorage

st.set_page_config(page_title="Dashboard | Alexandria", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 System Dashboard")

# --- System Stats ---
st.subheader("🖥️ Hardware Monitor")
col1, col2, col3 = st.columns(3)

cpu = psutil.cpu_percent()
ram = psutil.virtual_memory()
disk = psutil.disk_usage('.')

with col1:
    st.metric("CPU Usage", f"{cpu}%")
    st.progress(cpu / 100)

with col2:
    st.metric("RAM Usage", f"{ram.percent}%", f"{ram.used / (1024**3):.1f} GB / {ram.total / (1024**3):.1f} GB")
    st.progress(ram.percent / 100)

with col3:
    st.metric("Disk Usage", f"{disk.percent}%")
    st.progress(disk.percent / 100)

# --- Database Stats ---
st.markdown("---")
st.subheader("💾 Knowledge Base (LanceDB)")

try:
    storage = LanceDBStorage()
    count = storage.count()
    table_name = storage.table_name
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Chunks Indexed", count)
    with c2:
        st.metric("Table Name", table_name)
        
except Exception as e:
    st.error(f"Erro ao conectar ao LanceDB: {e}")

# --- Brain Stats ---
st.markdown("---")
st.subheader("🧠 Neural Network Stats")

mycelial = get_mycelial()
stats = mycelial.get_network_stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Observations", stats['total_observations'])
c2.metric("Active Connections", stats['active_connections'])
c3.metric("Density", f"{stats['density']:.6f}")
c4.metric("Max Strength", f"{stats['max_connection_strength']:.2f}")

# Top Connections
st.markdown("#### Top Neural Pathways")
top_conns = mycelial.get_strongest_connections(10)

if top_conns:
    st.table(top_conns)
else:
    st.info("Nenhuma conexão forte formada ainda.")
