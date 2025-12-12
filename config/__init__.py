import os
import torch

class Settings:
    # Caminhos
    # BASE_DIR = root of the project (parent of config folder)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INDEX_FILE = os.path.join(DATA_DIR, "knowledge.sfs")
    MAGIC_FOLDER = os.path.join(DATA_DIR, "magic_folder")
    
    # Causal Engine Paths
    CAUSAL_GRAPH_FILE = os.path.join(DATA_DIR, "causal_graph.json")
    LATENT_VARIABLES_FILE = os.path.join(DATA_DIR, "latent_variables.json")
    QUERY_LOGS_FILE = os.path.join(DATA_DIR, "query_logs.json")
    TOPOLOGY_FILE = os.path.join(DATA_DIR, "topology.json")
    
    # Modelos
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # Parâmetros Topológicos
    N_CLUSTERS = 256
    CHUNK_SIZE = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Causal Engine Parameters
    MIN_CO_OCCURRENCE = 3  # Mínimo de co-ocorrências para considerar causal
    LATENT_VARIABLE_THRESHOLD = 0.7  # Threshold para força de variável latente
    ORPHAN_CLUSTER_THRESHOLD = 0.1  # % de clusters isolados para ser considerado problema
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # === Hybrid Architecture Settings ===
    USE_HYBRID_MODE = True
    
    # AMD GPU no Windows tem suporte limitado PyTorch (CUDA é NVIDIA).
    # No Fedora com RX 580 (Polaris), ROCm pode ser instável para LLMs recentes.
    # Recomendado manter CPU float32 para estabilidade garantida.
    # Usaremos a CPU i9 que é muito forte, com float32 para velocidade.
    USE_LOCAL_LLM = False  # DESATIVADO POR PERFORMANCE/RECURSO (USER REQUEST)
    LOCAL_LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    LOCAL_LLM_DEVICE = "cpu"  # Forçar CPU para evitar erros de CUDA/AMD (Estável no Fedora)
    LOCAL_LLM_MAX_LENGTH = 512
    LOCAL_LLM_BATCH_SIZE = 1  # Batch 1 para menor latência
    LOCAL_LLM_THREADS = 8     # Usar 8 threads físicas do i9
    
    # Parâmetros Manifold (Otimização Geodesic 32d)
    MANIFOLD_DIM = 32  # Redução 384d -> 32d para cálculo geodésico rápido
    
    # Gemini (Expert Estratégico)
    GEMINI_MODEL = "gemini-2.0-flash"
    GEMINI_REFINEMENT_THRESHOLD = 0.7
    
    # Safety Parameters
    MAX_QUERY_LENGTH = 2000
    MAX_RESPONSE_TOKENS = 256

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        # Otimização PyTorch CPU
        torch.set_num_threads(self.LOCAL_LLM_THREADS)

settings = Settings()