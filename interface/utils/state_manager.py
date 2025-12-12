import streamlit as st
import sys
from pathlib import Path

# Add root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from core.reasoning.mycelial_reasoning import MycelialReasoning
from core.reasoning.neural_learner import V2Learner
from core.reasoning.abduction_engine import AbductionEngine
from sentence_transformers import SentenceTransformer
from config import settings

def init_session_state():
    """Initializes the global session state for the application."""
    
    # 1. Mycelial Reasoning (The Brain)
    if 'mycelial' not in st.session_state:
        st.session_state.mycelial = MycelialReasoning()
        
    # 2. Neural Learner (Lazy Load)
    if 'learner' not in st.session_state:
        st.session_state.learner = None
        
    # 3. Text Encoder (Lazy Load)
    if 'encoder' not in st.session_state:
        st.session_state.encoder = None
        
    # 4. Abduction Engine
    if 'abduction_engine' not in st.session_state:
        st.session_state.abduction_engine = AbductionEngine()

def get_learner():
    """Lazy loads the V2Learner model."""
    if st.session_state.learner is None:
        with st.spinner("🧠 Carregando Neural Learner (Monolith V13)..."):
            st.session_state.learner = V2Learner()
    return st.session_state.learner

def get_encoder():
    """Lazy loads the SentenceTransformer model."""
    if st.session_state.encoder is None:
        with st.spinner("📖 Carregando Encoder Textual..."):
            st.session_state.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return st.session_state.encoder

def get_mycelial():
    """Returns the MycelialReasoning instance."""
    if 'mycelial' not in st.session_state:
        init_session_state()
    return st.session_state.mycelial

def get_abduction_engine():
    """Returns the AbductionEngine instance."""
    if 'abduction_engine' not in st.session_state:
        init_session_state()
    return st.session_state.abduction_engine
