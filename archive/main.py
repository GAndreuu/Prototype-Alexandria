from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import uvicorn
import psutil
import numpy as np
from sklearn.decomposition import PCA
from core.topology_engine import TopologyEngine
from core.semantic_memory import SemanticFileSystem
from core.neural_learner import V2Learner
from core.oracle import NeuralOracle
from core.causal_reasoning import CausalEngine
from core.abduction_engine import AbductionEngine
from core.critic_agent import CriticAgent, CriticalAssessment, SystemFeedback
from core.action_agent import (
    ActionAgent, TestSimulator, EvidenceRegistrar, 
    ActionType, EvidenceType, create_action_agent_system
)
from config import settings
import os

# Configurar chave Gemini diretamente no ambiente
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="Prototype Alexandria")

# --- Estado Global (Singleton) ---
# Carregamos os modelos na memória ao iniciar
# Carregamos os modelos na memória ao iniciar
engine = TopologyEngine()
# Tentar carregar topologia salva
if os.path.exists(settings.TOPOLOGY_FILE):
    print(f"Carregando topologia de {settings.TOPOLOGY_FILE}...")
    engine.load_topology(settings.TOPOLOGY_FILE)
else:
    print("Nenhuma topologia salva encontrada. Necessário treinar.")
memory = SemanticFileSystem(engine, engine_encoder=engine)
# Tentar carregar índice salvo
if os.path.exists(settings.INDEX_FILE):
    print(f"Carregando índice de {settings.INDEX_FILE}...")
    memory.load_index(settings.INDEX_FILE)
else:
    print("Nenhum índice salvo encontrado. Começando do zero.")
oracle = NeuralOracle()
causal_engine = CausalEngine(engine, memory)
abduction_engine = AbductionEngine(settings.INDEX_FILE)
critic_agent = CriticAgent(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),  # Pode ser None para simulação
    sfs_path=settings.DATA_DIR,
    risk_tolerance=0.7
)

# === V12 - Action Agent: Capacidade de Intervenção no Mundo ===
action_agent, test_simulator, evidence_registrar = create_action_agent_system(
    sfs_instance=memory,
    sfs_path=settings.DATA_DIR
)

# --- Modelos de Dados ---
class QueryRequest(BaseModel):
    text: str
    mode: str = "hybrid"  # "local", "hybrid", "gemini"
    use_rag: bool = True  # Novo: Permite desligar a busca em livros

class IngestRequest(BaseModel):
    file_path: str
    type: str = "GEN"

class CausalQuery(BaseModel):
    text: str
    explain: bool = True

class AbductionRequest(BaseModel):
    max_hypotheses: int = 10

class CriticRequest(BaseModel):
    hypothesis_id: str
    hypothesis: dict
    supporting_evidence: list = []
    contradicting_evidence: list = []

class SystemAdjustmentRequest(BaseModel):
    temperature_adjustment: float = 0.0
    variance_adjustment: float = 0.0

# --- Modelos para Action Agent V12 ---

class ActionExecutionRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any]

class HypothesisTestRequest(BaseModel):
    hypothesis: Dict[str, Any]

class ParameterSimulationRequest(BaseModel):
    hypothesis_id: str
    parameter_name: str = "V11_BETA"
    parameter_values: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0]

class SecurityConfigRequest(BaseModel):
    allowed_apis: List[str] = []
    blocked_domains: List[str] = []

# --- Endpoints da ASI Base ---

@app.post("/train_topology")
async def train_topology(path: str):
    """
    Endpoint especial: Lê um arquivo grande para aprender os Conceitos.
    Deve ser rodado uma vez no início.
    """
    if not os.path.exists(path):
        raise HTTPException(404, "Arquivo de treino não encontrado")
    
    with open(path, 'r', encoding='utf-8') as f:
        # Pega uma amostra de 50k caracteres para treinar rápido
        sample_text = f.read(50000)
    
    chunks = [sample_text[i:i+256] for i in range(0, len(sample_text), 256)]
    vecs = engine.encode(chunks)
    
    # Ajustar clusters dinamicamente para evitar erro se tivermos poucos dados
    n_clusters = min(256, len(vecs))
    if n_clusters < 1:
        n_clusters = 1
        
    engine.train_manifold(vecs, n_clusters=n_clusters)
    
    # Salvar topologia treinada
    engine.save_topology(settings.TOPOLOGY_FILE)
    
    return {"status": "Topologia Treinada", "clusters": n_clusters}

@app.delete("/memory/file")
async def delete_file(file_id: str):
    """Remove um arquivo da memória"""
    try:
        if file_id in memory.indexed_files:
            del memory.indexed_files[file_id]
            # Salvar índice atualizado
            memory.save_index(settings.INDEX_FILE)
            return {"status": "Arquivo removido", "file_id": file_id}
        else:
            raise HTTPException(404, "Arquivo não encontrado")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ingest")
async def ingest_file(req: IngestRequest):
    """Adiciona um arquivo ao Cérebro (.sfs)"""
    if not engine.is_trained:
        raise HTTPException(400, "Treine a topologia primeiro! (/train_topology)")
    
    chunks_count = memory.index_file(req.file_path, req.type)
    
    # Auto-Resumo (Feature de Confiança)
    summary = "Resumo indisponível."
    try:
        # Pega os primeiros 3 chunks para ter uma ideia do conteúdo
        preview_chunks = memory.retrieve("resumo do conteúdo", limit=3)
        if preview_chunks:
            summary = oracle.synthesize(
                "Faça um resumo curto e empolgante sobre o que este arquivo trata.", 
                preview_chunks, 
                mode="local"
            )
    except Exception as e:
        print(f"Erro no auto-resumo: {e}")

    return {"status": "Indexado", "chunks": chunks_count, "auto_summary": summary}

@app.post("/query")
async def ask_oracle(req: QueryRequest):
    """
    Pergunta algo à ASI.
    Modos disponíveis:
    - 'local': Usa apenas TinyLlama (Rápido, Offline)
    - 'hybrid': TinyLlama gera rascunho + Gemini refina (Recomendado)
    - 'gemini': Usa apenas Gemini (Mais lento, alta qualidade)
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")

    evidence = []
    
    # 1. Recupera Contexto Real (Apenas se RAG estiver ativo)
    if req.use_rag:
        evidence = memory.retrieve(req.text)
        if not evidence:
            # Se for RAG e não achar nada, avisa. Se for chat puro, segue sem evidência.
            return {"answer": "Não tenho memórias sobre este tópico nos livros.", "evidence": []}
    
    # 2. Gera Resposta com Modo Selecionado
    # Se use_rag=False, evidence será [], o LLM responderá com seu conhecimento base.
    answer = oracle.synthesize(req.text, evidence, mode=req.mode)
    
    return {
        "answer": answer,
        "mode": req.mode,
        "evidence": [e['content'][:50]+"..." for e in evidence] # Debug
    }

# --- Endpoints do Causal Engine (V8) ---

@app.post("/causal/build_graph")
async def build_causal_graph():
    """
    Constrói o grafo causal analisando dependências entre clusters.
    
    Este endpoint:
    1. Analisa co-ocorrência de clusters no SFS
    2. Identifica sequências causais
    3. Mapeia dependências estruturais
    4. Constrói grafo causal consolidado
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    try:
        causal_graph = causal_engine.build_causal_graph()
        stats = causal_engine.get_statistics()
        
        return {
            "status": "Grafo Causal Construído",
            "clusters": len(causal_graph),
            "edges": sum(len(neighbors) for neighbors in causal_graph.values()),
            "statistics": stats,
            "causal_relationships": causal_graph
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao construir grafo causal: {str(e)}")

@app.post("/causal/discover_variables")
async def discover_variables():
    """Descobre variáveis latentes"""
    try:
        variables = causal_engine.discover_latent_variables()
        return variables
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/causal/identify_gaps")
async def identify_logic_gaps():
    """
    Identifica lacunas lógicas no grafo causal.
    
    Estas lacunas são candidatos para abdução (geração de hipóteses futuras).
    Inclui clusters órfãos, cadeias quebradas e contradições aparentes.
    """
    if not causal_engine.causal_graph:
        raise HTTPException(400, "Construa o grafo causal primeiro! (/causal/build_graph)")
    
    try:
        gaps = causal_engine.identify_logic_gaps()
        
        return {
            "status": "Lacunas Lógicas Identificadas",
            "count": len(gaps),
            "gaps": gaps
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao identificar lacunas: {str(e)}")

@app.post("/causal/explain")
async def explain_causality(req: CausalQuery):
    """
    Explica a causalidade por trás de uma consulta específica.
    
    Retorna:
    - Causas diretas do conceito
    - Efeitos diretos
    - Variáveis latentes relevantes
    - Explicação textual da causalidade
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    if not causal_engine.causal_graph:
        raise HTTPException(400, "Construa o grafo causal primeiro! (/causal/build_graph)")
    
    try:
        explanation = causal_engine.explain_causality(req.text)
        
        return explanation
    except Exception as e:
        raise HTTPException(500, f"Erro ao explicar causalidade: {str(e)}")

@app.get("/causal/path/{source_cluster}/{target_cluster}")
async def get_causal_path(source_cluster: int, target_cluster: int):
    """
    Encontra caminho causal entre dois clusters específicos.
    
    Returns None se não houver caminho conectando os clusters.
    """
    if not causal_engine.causal_graph:
        raise HTTPException(400, "Construa o grafo causal primeiro! (/causal/build_graph)")
    
    if source_cluster < 0 or source_cluster >= 256 or target_cluster < 0 or target_cluster >= 256:
        raise HTTPException(400, "Clusters devem estar entre 0-255")
    
    try:
        path = causal_engine.get_causal_path(source_cluster, target_cluster)
        
        return {
            "source_cluster": source_cluster,
            "target_cluster": target_cluster,
            "path": path,
            "path_exists": path is not None,
            "path_length": len(path) if path else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao encontrar caminho causal: {str(e)}")

@app.get("/causal/statistics")
async def get_causal_statistics():
    """
    Retorna estatísticas do grafo causal.
    
    Métricas:
    - Total de clusters conectados
    - Total de arestas causais
    - Variáveis latentes descobertas
    - Razão de conectividade
    """
    if not causal_engine.causal_graph:
        raise HTTPException(400, "Construa o grafo causal primeiro! (/causal/build_graph)")
    
    try:
        stats = causal_engine.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter estatísticas: {str(e)}")

@app.get("/causal/graphviz")
async def get_causal_graphviz():
    """Retorna representação DOT do grafo para visualização"""
    try:
        # Gerar DOT simples baseado nos clusters
        stats = engine.get_topology_stats()
        if not stats.get("is_trained"):
            return {"dot": "digraph G { label=\"Cérebro não treinado\"; }"}
            
        dot = "digraph G {\n"
        dot += "  bgcolor=\"#0e1117\";\n"
        dot += "  node [style=filled, fillcolor=\"#21262d\", fontcolor=\"#c9d1d9\", color=\"#30363d\"];\n"
        dot += "  edge [color=\"#58a6ff\"];\n"
        
        # Adicionar nós (clusters)
        centers = engine.get_cluster_centers()
        if centers is not None:
            # Mostrar apenas os top 20 clusters mais densos para não poluir
            labels = engine.cluster_labels
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1][:20]
            
            for i in sorted_indices:
                cluster_id = unique[i]
                count = counts[i]
                dot += f"  c{cluster_id} [label=\"Cluster {cluster_id}\\n({count} itens)\"];\n"
                
            # Adicionar arestas fictícias para visualização de "conexão" (simulação de topologia)
            # Em um sistema real, isso viria da matriz de adjacência do manifold
            for i in range(len(sorted_indices) - 1):
                dot += f"  c{unique[sorted_indices[i]]} -> c{unique[sorted_indices[i+1]]};\n"
                
        dot += "}"
        return {"dot": dot}
    except Exception as e:
        print(f"Erro Graphviz: {e}")
        return {"dot": "digraph G { label=\"Erro na visualização\"; }"}

@app.get("/system/stats")
async def get_system_stats():
    """Retorna estatísticas do sistema"""
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            "cpu_percent": cpu_usage,
            "ram_percent": ram.percent,
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "files_indexed": len(memory.indexed_files)
        }
    except Exception as e:
        print(f"Erro Stats: {e}")
        return {}

@app.get("/causal/status")
async def get_causal_status():
    """
    Status geral do Causal Engine.
    
    Indica quais componentes estão prontos e quais precisam ser executados.
    """
    return {
        "topology_trained": engine.is_trained,
        "causal_graph_built": bool(causal_engine.causal_graph),
        "latent_variables_found": bool(causal_engine.latent_variables),
        "index_file_exists": os.path.exists(settings.INDEX_FILE),
        "ready_for_causal_analysis": engine.is_trained and os.path.exists(settings.INDEX_FILE)
    }

# --- Endpoints do Abduction Engine (V9) ---

@app.post("/abduction/detect_gaps")
async def detect_knowledge_gaps():
    """
    Detecta lacunas no conhecimento causal.
    
    Este endpoint identifica:
    - Clusters órfãos (sem conexões)
    - Cadeias quebradas
    - Conexões óbvias faltantes
    - Contradições potenciais
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    try:
        gaps = abduction_engine.detect_knowledge_gaps()
        
        return {
            "status": "Lacunas Detectadas",
            "count": len(gaps),
            "gaps": [
                {
                    "gap_id": gap.gap_id,
                    "gap_type": gap.gap_type,
                    "description": gap.description,
                    "affected_clusters": gap.affected_clusters,
                    "priority_score": gap.priority_score,
                    "detected_at": gap.detected_at.isoformat()
                }
                for gap in gaps
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao detectar lacunas: {str(e)}")

@app.post("/abduction/generate_hypotheses")
async def generate_hypotheses(req: AbductionRequest):
    """
    Gera hipóteses automaticamente para preencher lacunas identificadas.
    
    Este endpoint:
    1. Analisa lacunas detectadas
    2. Gera hipóteses causais baseadas em padrões
    3. Calcula scores de confiança
    4. Armazena hipóteses para validação
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    try:
        hypotheses = abduction_engine.generate_hypotheses(req.max_hypotheses)
        
        return {
            "status": "Hipóteses Geradas",
            "count": len(hypotheses),
            "hypotheses": [
                {
                    "id": hyp.id,
                    "source_cluster": hyp.source_cluster,
                    "target_cluster": hyp.target_cluster,
                    "hypothesis_text": hyp.hypothesis_text,
                    "confidence_score": hyp.confidence_score,
                    "evidence_strength": hyp.evidence_strength,
                    "validation_status": hyp.validation_status,
                    "created_at": hyp.created_at.isoformat()
                }
                for hyp in hypotheses
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar hipóteses: {str(e)}")

@app.post("/abduction/validate_hypothesis")
async def validate_hypothesis(hypothesis_id: str):
    """
    Valida uma hipótese específica usando dados do SFS.
    
    Executa testes de:
    - Coerência semântica
    - Co-ocorrência em documentos
    - Padrões sequenciais
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    try:
        is_valid = abduction_engine.validate_hypothesis(hypothesis_id)
        hypothesis = abduction_engine.hypotheses.get(hypothesis_id)
        
        if not hypothesis:
            raise HTTPException(404, "Hipótese não encontrada")
        
        return {
            "status": "Validação Concluída",
            "hypothesis_id": hypothesis_id,
            "is_valid": is_valid,
            "validation_status": hypothesis.validation_status,
            "confidence_score": hypothesis.confidence_score,
            "validated_at": hypothesis.validated_at.isoformat() if hypothesis.validated_at else None
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao validar hipótese: {str(e)}")

@app.post("/abduction/expand_knowledge")
async def expand_knowledge_graph():
    """
    Expande o grafo causal com hipóteses validadas.
    
    Este endpoint:
    1. Identifica hipóteses validadas
    2. Adiciona conexões ao grafo causal
    3. Calcula estatísticas da expansão
    """
    if not causal_engine.causal_graph:
        raise HTTPException(400, "Construa o grafo causal primeiro! (/causal/build_graph)")
    
    try:
        # Obter hipóteses validadas
        validated_ids = [
            hyp_id for hyp_id, hyp in abduction_engine.hypotheses.items()
            if hyp.validation_status == "validated"
        ]
        
        if not validated_ids:
            return {
                "status": "Nenhuma Hipótese Validada",
                "message": "Valide hipóteses primeiro antes de expandir o grafo"
            }
        
        expansion_stats = abduction_engine.expand_knowledge_graph(validated_ids)
        
        return {
            "status": "Conhecimento Expandido",
            "validated_hypotheses": len(validated_ids),
            "expansion_statistics": expansion_stats
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao expandir conhecimento: {str(e)}")

@app.post("/abduction/run_cycle")
async def run_abduction_cycle(req: AbductionRequest):
    """
    Executa um ciclo completo de abdução.
    
    Pipeline completo:
    1. Detectar lacunas
    2. Gerar hipóteses
    3. Validar hipóteses
    4. Expandir conhecimento
    5. Retornar estatísticas
    """
    if not engine.is_trained:
        raise HTTPException(400, "Cérebro não inicializado.")
    
    try:
        cycle_stats = abduction_engine.run_abduction_cycle(req.max_hypotheses)
        
        return {
            "status": "Ciclo de Abdução Concluído",
            "cycle_statistics": cycle_stats
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao executar ciclo: {str(e)}")

@app.get("/abduction/report")
async def get_abduction_report():
    """
    Gera relatório completo do estado do motor de abdução.
    
    Inclui:
    - Estatísticas gerais
    - Métricas de confiança
    - Tendências de evolução
    - Histórico recente
    """
    try:
        report = abduction_engine.get_abduction_report()
        
        return {
            "status": "Relatório Gerado",
            "report": report
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar relatório: {str(e)}")

@app.get("/abduction/status")
async def get_abduction_status():
    """
    Status geral do Abduction Engine.
    
    Indica o estado de readiness para diferentes operações.
    """
    return {
        "abduction_engine_ready": True,
        "hypotheses_count": len(abduction_engine.hypotheses),
        "knowledge_gaps_count": len(abduction_engine.knowledge_gaps),
        "validation_tests_count": len(abduction_engine.validation_tests),
        "ready_for_gap_detection": engine.is_trained,
        "ready_for_hypothesis_generation": engine.is_trained and len(abduction_engine.knowledge_gaps) > 0,
        "ready_for_validation": len(abduction_engine.hypotheses) > 0,
        "ready_for_expansion": len(abduction_engine.hypotheses) > 0
    }

# === V10 - Consciência Interativa: Agente Crítico Sênior ===

@app.post("/critic/assess_hypothesis")
async def assess_hypothesis(req: CriticRequest):
    """
    Avalia uma hipótese gerada pelo V9 usando o Agente Crítico Sênior (V10).
    
    Este endpoint:
    1. Recebe a hipótese do V9
    2. Coleta evidências de suporte e contradição do SFS
    3. Usa Gemini para análise crítica
    4. Calcula Pontuação de Risco
    5. Retorna avaliação completa com recomendações
    """
    try:
        assessment = await critic_agent.assess_hypothesis(
            hypothesis=req.hypothesis,
            supporting_evidence=req.supporting_evidence,
            contradicting_evidence=req.contradicting_evidence
        )
        
        return {
            "status": "Avaliação Concluída",
            "assessment": {
                "hypothesis_id": assessment.hypothesis_id,
                "truth_score": assessment.truth_score,
                "truth_category": assessment.truth_category.value,
                "risk_level": assessment.risk_level.value,
                "confidence_intervals": assessment.confidence_intervals,
                "evidence_quality": assessment.evidence_quality,
                "contradiction_strength": assessment.contradiction_strength,
                "reasoning_coherence": assessment.reasoning_coherence,
                "supporting_facts": assessment.supporting_facts,
                "contradicting_facts": assessment.contradicting_facts,
                "gaps_in_evidence": assessment.gaps_in_evidence,
                "recommendation": assessment.recommendation,
                "suggested_adjustments": assessment.suggested_adjustments,
                "assessed_at": assessment.assessed_at.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao avaliar hipótese: {str(e)}")

@app.get("/critic/system_feedback")
async def get_system_feedback():
    """
    Gera feedback do sistema para auto-regulagem (V10).
    
    Fornece:
    - Métricas de performance
    - Indicadores de bias
    - Recomendações de ajuste
    - Análise de contradições comuns
    """
    try:
        feedback = await critic_agent.get_system_feedback()
        
        return {
            "status": "Feedback do Sistema",
            "feedback": {
                "timestamp": feedback.timestamp.isoformat(),
                "total_assessments": feedback.total_assessments,
                "approval_rate": feedback.approval_rate,
                "average_truth_score": feedback.average_truth_score,
                "common_contradictions": feedback.common_contradictions,
                "system_bias_indicators": feedback.system_bias_indicators,
                "recommended_temperature_adjustment": feedback.recommended_temperature_adjustment,
                "recommended_variance_adjustment": feedback.recommended_variance_adjustment
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar feedback: {str(e)}")

@app.post("/critic/apply_adjustments")
async def apply_system_adjustments(req: SystemAdjustmentRequest):
    """
    Aplica ajustes automáticos aos parâmetros do sistema (V10).
    
    Permite:
    - Ajuste de temperatura do TinyLlama
    - Ajuste de variância do Manifold
    - Auto-regulagem baseada em feedback
    """
    try:
        critic_agent.apply_system_adjustments(
            temperature_adjustment=req.temperature_adjustment,
            variance_adjustment=req.variance_adjustment
        )
        
        return {
            "status": "Ajustes Aplicados",
            "adjustments": {
                "temperature_adjustment": req.temperature_adjustment,
                "variance_adjustment": req.variance_adjustment,
                "applied_at": "agora",
                "total_adjustments": critic_agent.system_metrics["temperature_adjustments"]
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao aplicar ajustes: {str(e)}")

@app.get("/critic/assessment_history")
async def get_assessment_history(limit: int = 50):
    """
    Retorna histórico de avaliações do Critic Agent (V10).
    
    Inclui:
    - Últimas avaliações realizadas
    - Distribuição de scores
    - Tendências de aprovação/rejeição
    """
    try:
        recent_assessments = critic_agent.assessment_history[-limit:]
        
        return {
            "status": "Histórico Recuperado",
            "history": {
                "total_assessments": len(critic_agent.assessment_history),
                "showing_recent": len(recent_assessments),
                "assessments": [
                    {
                        "hypothesis_id": assessment.hypothesis_id,
                        "truth_score": assessment.truth_score,
                        "risk_level": assessment.risk_level.value,
                        "recommendation": assessment.recommendation,
                        "assessed_at": assessment.assessed_at.isoformat()
                    }
                    for assessment in recent_assessments
                ]
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao recuperar histórico: {str(e)}")

@app.get("/critic/export_report")
async def export_critic_report(format_type: str = "json"):
    """
    Exporta relatório completo do Agente Crítico (V10).
    
    Formatos disponíveis:
    - "json": Relatório estruturado em JSON
    - "markdown": Relatório legível em Markdown
    """
    try:
        if format_type not in ["json", "markdown"]:
            raise HTTPException(400, "Formato não suportado. Use 'json' ou 'markdown'.")
        
        report = critic_agent.export_assessment_report(format_type)
        
        return {
            "status": "Relatório Exportado",
            "format": format_type,
            "report": report,
            "generated_at": "2025-11-22T14:38:46"
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao exportar relatório: {str(e)}")

@app.get("/critic/status")
async def get_critic_status():
    """
    Status geral do Agente Crítico Sênior (V10).
    
    Indica:
    - Estado de readiness do CriticAgent
    - Métricas de sistema
    - Indicadores de bias
    - Recomendações ativas
    """
    try:
        return {
            "critic_agent_ready": True,
            "system_metrics": critic_agent.system_metrics,
            "bias_indicators": critic_agent.bias_indicators,
            "total_assessments": len(critic_agent.assessment_history),
            "recent_temperature_history": critic_agent.temperature_history[-10:] if critic_agent.temperature_history else [],
            "gemini_api_configured": critic_agent.gemini_api_key is not None,
            "risk_tolerance": critic_agent.risk_tolerance,
            "capabilities": [
                "Hypothesis Assessment",
                "Risk Scoring", 
                "Bias Detection",
                "Auto-Adjustment",
                "System Feedback",
                "Report Generation"
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao verificar status: {str(e)}")

# === V12 - Action Agent: Endpoints de Execução ===

@app.post("/action/execute")
async def execute_action(req: ActionExecutionRequest):
    """
    Executa uma ação segura no backend usando o Action Agent.
    
    Tipos de ação suportados:
    - "api_call": Chamada de API com validação
    - "parameter_adjustment": Ajuste de parâmetro do sistema
    - "model_retrain": Re-treinamento simulado de modelo
    - "data_generation": Geração de dados sintéticos
    - "system_config_change": Mudança de configuração
    - "simulation_run": Execução de simulação customizada
    """
    try:
        action_type = ActionType(req.action_type)
        result = action_agent.execute_action(
            action_type=action_type,
            parameters=req.parameters
        )
        
        return {
            "status": "Ação Executada",
            "action_id": result.action_id,
            "action_type": result.action_type.value,
            "execution_status": result.status.value,
            "duration": result.duration,
            "result_data": result.result_data,
            "error_message": result.error_message,
            "evidence_generated": result.evidence_generated,
            "evidence_type": result.evidence_type.value if result.evidence_type else None
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao executar ação: {str(e)}")

@app.post("/action/test_hypothesis")
async def test_hypothesis(req: HypothesisTestRequest):
    """
    Testa uma hipótese gerada pelo V9 usando o Action Agent.
    
    Este endpoint:
    1. Recebe a hipótese do V9
    2. Determina a ação de teste apropriada
    3. Executa teste via ActionAgent
    4. Avalia resultado como suporte/refutação
    5. Registra evidência no SFS
    """
    try:
        test_result = action_agent.test_hypothesis(req.hypothesis)
        
        return {
            "status": "Teste de Hipótese Concluído",
            "hypothesis_id": test_result.hypothesis_id,
            "test_status": test_result.result.status.value if test_result.result else "failed",
            "evidence_type": test_result.result.evidence_type.value if test_result.result and test_result.result.evidence_type else "neutral",
            "duration": test_result.result.duration if test_result.result else 0,
            "evidence_registered": test_result.evidence_registered,
            "test_details": {
                "hypothesis_text": test_result.hypothesis_text,
                "source_cluster": test_result.source_cluster,
                "target_cluster": test_result.target_cluster,
                "test_action": test_result.test_action.value,
                "test_parameters": test_result.test_parameters
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao testar hipótese: {str(e)}")

@app.post("/action/simulate_parameter")
async def simulate_parameter_change(req: ParameterSimulationRequest):
    """
    Simula mudanças em parâmetros do V11 Vision Encoder e gera logs de acurácia.
    
    Testa múltiplos valores de um parâmetro específico e determina qual
    valor otimiza a performance do modelo.
    """
    try:
        # Criar hipótese para o teste
        hypothesis = {
            "id": req.hypothesis_id,
            "test_parameters": {
                "parameter": req.parameter_name,
                "values": req.parameter_values
            },
            "expected_outcome": {"optimization_goal": "accuracy"},
            "validation_criteria": {"min_improvement": 0.05}
        }
        
        if req.parameter_name == "V11_BETA":
            simulation_result = test_simulator.simulate_v11_parameter_test(hypothesis)
        else:
            raise ValueError(f"Parâmetro não suportado para simulação: {req.parameter_name}")
        
        return {
            "status": "Simulação Concluída",
            "simulation_id": req.hypothesis_id,
            "parameter_tested": req.parameter_name,
            "simulation_results": simulation_result,
            "best_value": simulation_result.get("best_value"),
            "best_accuracy": simulation_result.get("best_accuracy"),
            "improvement": simulation_result.get("best_accuracy") - 0.8  # Baseline assumido
        }
    except Exception as e:
        raise HTTPException(500, f"Erro na simulação: {str(e)}")

@app.get("/action/statistics")
async def get_action_statistics():
    """
    Retorna estatísticas completas do Action Agent V12.
    
    Inclui:
    - Estatísticas de execução de ações
    - Resultados de testes de hipóteses
    - Métricas de simulações
    - Estatísticas de evidências registradas
    """
    try:
        action_stats = action_agent.get_test_statistics()
        simulation_report = test_simulator.get_simulation_report()
        evidence_stats = evidence_registrar.get_evidence_statistics()
        security_log = action_agent.security_controller.get_audit_log()
        
        return {
            "status": "Estatísticas Recuperadas",
            "action_statistics": action_stats,
            "simulation_statistics": simulation_report,
            "evidence_statistics": evidence_stats,
            "security_audit": {
                "total_logged_actions": len(security_log),
                "recent_actions": security_log[-10:]  # Últimas 10
            },
            "system_health": {
                "action_agent_ready": True,
                "test_simulator_ready": True,
                "evidence_registrar_ready": True,
                "security_level": action_agent.security_level
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter estatísticas: {str(e)}")

@app.get("/action/security/status")
async def get_security_status():
    """
    Status de segurança do Action Agent.
    
    Mostra:
    - APIs permitidas
    - Rate limits ativos
    - Domínios bloqueados
    - Log de auditoria recente
    """
    try:
        security = action_agent.security_controller
        
        return {
            "status": "Status de Segurança",
            "allowed_apis": security.allowed_apis,
            "blocked_domains": list(security.blocked_domains),
            "rate_limits_active": len(security.rate_limits),
            "audit_log_entries": len(security.audit_log),
            "recent_audit_log": security.get_audit_log(20),
            "security_level": action_agent.security_level
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao verificar segurança: {str(e)}")

@app.post("/action/security/configure")
async def configure_security(req: SecurityConfigRequest):
    """
    Configura parâmetros de segurança do Action Agent.
    
    Permite:
    - Definir lista de APIs permitidas
    - Bloquear domínios específicos
    - Ajustar níveis de segurança
    """
    try:
        # Atualizar APIs permitidas
        if req.allowed_apis:
            action_agent.security_controller.allowed_apis = req.allowed_apis
            os.environ["ALLOWED_APIS"] = ",".join(req.allowed_apis)
        
        # Atualizar domínios bloqueados
        if req.blocked_domains:
            action_agent.security_controller.blocked_domains = set(req.blocked_domains)
        
        return {
            "status": "Segurança Configurada",
            "allowed_apis": action_agent.security_controller.allowed_apis,
            "blocked_domains": list(action_agent.security_controller.blocked_domains),
            "configured_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao configurar segurança: {str(e)}")

@app.get("/action/test_history")
async def get_test_history(limit: int = 50):
    """
    Histórico completo de testes executados pelo Action Agent.
    
    Retorna:
    - Lista de hipóteses testadas
    - Resultados de cada teste
    - Estatísticas de performance
    - Evidências geradas
    """
    try:
        # Obter histórico de testes
        test_history = []
        for hyp_id, test_result in action_agent.test_hypotheses.items():
            test_entry = {
                "hypothesis_id": hyp_id,
                "hypothesis_text": test_result.hypothesis_text,
                "test_action": test_result.test_action.value,
                "test_status": test_result.result.status.value if test_result.result else "pending",
                "evidence_type": test_result.result.evidence_type.value if test_result.result and test_result.result.evidence_type else "none",
                "executed_at": test_result.executed_at.isoformat() if test_result.executed_at else None,
                "duration": test_result.result.duration if test_result.result else 0,
                "evidence_registered": test_result.evidence_registered
            }
            test_history.append(test_entry)
        
        # Ordenar por data (mais recentes primeiro)
        test_history.sort(key=lambda x: x["executed_at"] or "", reverse=True)
        
        return {
            "status": "Histórico Recuperado",
            "total_tests": len(test_history),
            "showing_recent": min(len(test_history), limit),
            "test_history": test_history[:limit],
            "summary": {
                "supporting_evidence": len([t for t in test_history if t["evidence_type"] == "supporting"]),
                "contradicting_evidence": len([t for t in test_history if t["evidence_type"] == "contradicting"]),
                "neutral_evidence": len([t for t in test_history if t["evidence_type"] == "neutral"]),
                "inconclusive": len([t for t in test_history if t["evidence_type"] == "none"])
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter histórico: {str(e)}")

@app.get("/system/status")
async def system_status():
    """
    Status geral do sistema Prototype Alexandria.
    """
    return {
        "system": "Prototype Alexandria",
        "version": "V12 (Action Agent - Capacidade de Intervenção)",
        "components": {
            "topology_engine": {
                "status": "ready" if engine.is_trained else "needs_training",
                "model": settings.EMBEDDING_MODEL,
                "clusters": settings.N_CLUSTERS
            },
            "semantic_file_system": {
                "status": "ready",
                "index_file": settings.INDEX_FILE,
                "index_exists": os.path.exists(settings.INDEX_FILE)
            },
            "neural_oracle": {
                "status": "ready",
                "local_model": settings.LLM_MODEL,
                "hybrid_mode": True
            },
            "causal_engine": {
                "status": "ready" if causal_engine.causal_graph else "needs_graph",
                "causal_graph": bool(causal_engine.causal_graph),
                "latent_variables": bool(causal_engine.latent_variables)
            },
            "abduction_engine": {
                "status": "ready" if engine.is_trained else "needs_training",
                "hypotheses_count": len(abduction_engine.hypotheses),
                "knowledge_gaps_count": len(abduction_engine.knowledge_gaps),
                "cycle_capable": engine.is_trained and causal_engine.causal_graph is not None
            },
            "critic_agent": {
                "status": "ready",
                "total_assessments": len(critic_agent.assessment_history),
                "gemini_api_configured": critic_agent.gemini_api_key is not None,
                "risk_tolerance": critic_agent.risk_tolerance,
                "bias_indicators": critic_agent.bias_indicators
            },
            "action_agent": {
                "status": "ready",
                "total_tests": len(action_agent.test_hypotheses),
                "security_level": action_agent.security_level,
                "evidence_registered": evidence_registrar.get_evidence_statistics()["total_evidence"],
                "simulations_completed": len(test_simulator.simulation_history)
            },
            "v2_learner": {
                "status": "active",
                "model_loaded": action_agent.v2_learner.is_loaded
            }
        },
        "endpoints": {
            "training": ["/train_topology"],
            "ingestion": ["/ingest"],
            "querying": ["/query"],
            "causal_analysis": [
                "/causal/build_graph",
                "/causal/discover_variables", 
                "/causal/identify_gaps",
                "/causal/explain",
                "/causal/path/{source}/{target}",
                "/causal/statistics",
                "/causal/status"
            ],
            "abduction": [
                "/abduction/detect_gaps",
                "/abduction/generate_hypotheses",
                "/abduction/validate_hypothesis",
                "/abduction/expand_knowledge",
                "/abduction/run_cycle",
                "/abduction/report",
                "/abduction/status"
            ],
            "critic_v10": [
                "/critic/assess_hypothesis",
                "/critic/system_feedback",
                "/critic/apply_adjustments",
                "/critic/assessment_history",
                "/critic/export_report",
                "/critic/status"
            ],
            "action_v12": [
                "/action/execute",
                "/action/test_hypothesis",
                "/action/simulate_parameter",
                "/action/statistics",
                "/action/security/status",
                "/action/security/configure",
                "/action/security/configure",
                "/action/test_history"
            ],
            "learning_v13": [
                "/learning/status",
                "/learning/trigger"
            ]
        }
    }

# === V13 - Self-Feeding: V2 Learner Endpoints ===

@app.get("/learning/status")
async def get_learning_status():
    """Retorna status do módulo de aprendizado V2"""
    try:
        learner = action_agent.v2_learner
        return {
            "status": "active" if learner.is_loaded else "inactive",
            "model_path": str(learner.model_path),
            "device": learner.device,
            "is_loaded": learner.is_loaded
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter status de aprendizado: {str(e)}")

@app.post("/learning/trigger")
async def trigger_learning(vectors: List[List[float]]):
    """Gatilho manual para aprendizado"""
    try:
        result = action_agent.execute_action(
            ActionType.INTERNAL_LEARNING,
            {"vectors": vectors}
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Erro ao acionar aprendizado: {str(e)}")

# === V14 - Deep Analysis & Visualization ===

@app.get("/visualization/manifold_data")
async def get_manifold_data(limit: int = 2000):
    """
    Retorna dados do manifold reduzidos para visualização 3D/2D.
    Usa PCA para projetar vetores 384D em 3D.
    """
    try:
        points = []
        vectors = []
        metadata_list = []
        
        # 1. Coletar vetores de todos os arquivos indexados
        for file_id, file_meta in memory.indexed_files.items():
            for chunk in file_meta.get('indexed_chunks', []):
                if 'vector' in chunk:
                    vectors.append(chunk['vector'])
                    metadata_list.append({
                        "id": chunk['chunk_id'],
                        "content": chunk['content'][:100] + "...",
                        "modality": file_meta.get('modalidade', 'UNKNOWN'),
                        "source": os.path.basename(file_meta.get('file_path', '?')),
                        "cluster": chunk.get('cluster', 0) # Se tiver cluster salvo
                    })
                    if len(vectors) >= limit:
                        break
            if len(vectors) >= limit:
                break
        
        if not vectors:
            return {"points": []}
            
        # 2. Redução de Dimensionalidade (PCA)
        # Converter para numpy
        X = np.array(vectors)
        
        # Se tiver menos de 3 pontos, não dá pra fazer PCA 3D direito, mas o sklearn lida
        n_components = min(3, len(X), X.shape[1])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # 3. Formatar resposta
        for i, coords in enumerate(X_reduced):
            point = {
                "x": float(coords[0]),
                "y": float(coords[1]) if n_components > 1 else 0.0,
                "z": float(coords[2]) if n_components > 2 else 0.0,
                **metadata_list[i]
            }
            points.append(point)
            
        return {
            "status": "success",
            "count": len(points),
            "points": points,
            "explained_variance": pca.explained_variance_ratio_.tolist()
        }
        
    except Exception as e:
        print(f"Erro na visualização: {e}")
        raise HTTPException(500, f"Erro ao gerar visualização: {str(e)}")

@app.get("/visualization/evolution_stats")
async def get_evolution_stats():
    """Retorna histórico de evolução do sistema (Loss, Métricas)"""
    try:
        history = action_agent.v2_learner.history
        return {
            "status": "success",
            "history_points": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter estatísticas: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)