"""
Bridge Agent Module
Responsible for identifying knowledge gaps and planning bridge acquisition.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class KnowledgeGap:
    """Representa um 'gap' detectado pelo Abduction Engine."""
    gap_id: str

    # Conceitos extremos da lacuna (podem ser IDs ou descrições)
    source_concept: str
    target_concept: str

    # Embeddings contínuos (no espaço do Topology Engine)
    source_vec: np.ndarray  # shape [384] ou [d_model]
    target_vec: np.ndarray

    # Códigos discretos do VQ-VAE (micélio)
    source_codes: List[int]  # ex: [h1,h2,h3,h4]
    target_codes: List[int]

    # Infos semânticas / de contexto
    context_tags: List[str] = field(default_factory=list)
    relation_type: str = "unknown"  # ex: "missing_mechanism", "missing_application"

    # Opcional: score de importância do gap
    importance: float = 1.0

@dataclass
class BridgeSpec:
    """Descrição simbólica da 'ponte ideal' que estamos buscando."""
    domain: str                  # ex: "causal_inference"
    missing_piece: str           # ex: "algorithmic_implementation"
    formalism: str               # ex: "mathematical", "empirical", "survey"
    application_context: str     # ex: "reinforcement_learning"
    extra_keywords: List[str] = field(default_factory=list)

@dataclass
class BridgeRequest:
    """
    Pedido concreto que o sistema faz ao 'mundo' para procurar uma ponte:
    - query vetorial (p/ vector DB externo ou interno)
    - query simbólica/textual (p/ APIs tipo arXiv / SemanticScholar / etc.)
    """
    gap_id: str

    semantic_query_vec: np.ndarray       # vetor "ideal" da ponte
    bridge_spec: BridgeSpec             # descrição simbólica
    text_query: str                     # string de busca
    filters: Dict[str, str] = field(default_factory=dict)  # ex: {"year_from": "2018"}

@dataclass
class BridgeCandidate:
    """
    Resultado de uma busca de ponte: um paper/arquivo que pode ajudar a fechar o gap.
    """
    doc_id: str
    title: str
    abstract: str
    source: str  # ex: "local_corpus", "arxiv", "semantic_scholar"

    embedding: np.ndarray
    similarity_to_bridge: float
    similarity_to_source: float
    similarity_to_target: float
    novelty_score: float  # quão novo isso é comparado à memória atual

    # Score final que mistura tudo
    final_score: float

    # Opcional: metadados extras
    metadata: Dict = field(default_factory=dict)


def build_bridge_vector(
    source_vec: np.ndarray,
    target_vec: np.ndarray,
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.2,
) -> np.ndarray:
    """
    Cria um 'embedding ideal' da ponte:
        bridge_vec ≈ combinação de origem, destino e direção (target - source).
    """
    assert source_vec.shape == target_vec.shape, "Source e target precisam ter mesma dimensão"

    direction = target_vec - source_vec
    bridge_vec = w1 * source_vec + w2 * target_vec + w3 * direction

    # Normaliza pra norma 1 (ajuda em similarity)
    norm = np.linalg.norm(bridge_vec) + 1e-8
    return bridge_vec / norm


def infer_domain_from_tags(tags: List[str]) -> str:
    # TODO: ligar com ontologias reais do Alexandria
    if any("causal" in t for t in tags):
        return "causal_inference"
    if any("reinforcement" in t for t in tags):
        return "reinforcement_learning"
    if any("bayesian" in t for t in tags):
        return "bayesian_inference"
    return "unknown"


def infer_missing_piece(relation_type: str) -> str:
    if relation_type == "missing_mechanism":
        return "mechanism"
    if relation_type == "missing_application":
        return "application"
    if relation_type == "missing_theory":
        return "theoretical_justification"
    return "unspecified"


def build_bridge_spec(gap: KnowledgeGap) -> BridgeSpec:
    domain = infer_domain_from_tags(gap.context_tags)
    missing_piece = infer_missing_piece(gap.relation_type)

    # Chutes sensatos de default: você pode sofisticar isso depois
    if missing_piece in ["mechanism", "theoretical_justification"]:
        formalism = "mathematical"
    elif missing_piece == "application":
        formalism = "empirical"
    else:
        formalism = "unspecified"

    # Exemplo: tentar descobrir contexto de aplicação a partir das tags
    if any("rl" in t or "reinforcement" in t for t in gap.context_tags):
        app_ctx = "reinforcement_learning"
    elif any("health" in t or "clinic" in t for t in gap.context_tags):
        app_ctx = "healthcare"
    else:
        app_ctx = "generic"

    extra_keywords = [gap.source_concept, gap.target_concept] + gap.context_tags

    return BridgeSpec(
        domain=domain,
        missing_piece=missing_piece,
        formalism=formalism,
        application_context=app_ctx,
        extra_keywords=extra_keywords,
    )


def build_text_query(spec: BridgeSpec) -> str:
    """
    Constrói uma query textual estilo arXiv/Scholar a partir do BridgeSpec.
    Ex: "causal inference" AND "reinforcement learning" AND algorithm AND proof
    """
    terms = []

    if spec.domain != "unknown":
        terms.append(f"\"{spec.domain.replace('_', ' ')}\"")

    if spec.application_context not in ["generic", ""]:
        terms.append(f"\"{spec.application_context.replace('_', ' ')}\"")

    if spec.missing_piece == "mechanism":
        terms.append("mechanism")
    elif spec.missing_piece == "theoretical_justification":
        terms.append("proof OR theorem OR guarantee")
    elif spec.missing_piece == "application":
        terms.append("application OR case study")

    if spec.formalism == "mathematical":
        terms.append("theoretical OR formal")
    elif spec.formalism == "empirical":
        terms.append("experiment OR empirical")

    # Extra keywords (conceitos extremos + tags)
    for kw in spec.extra_keywords:
        k = kw.strip()
        if not k:
            continue
        terms.append(f"\"{k}\"")

    # Junta tudo num AND meio tosco, mas funcional
    if not terms:
        return ""  # fallback, você pode definir um default melhor

    return " AND ".join(terms)


def plan_bridge_acquisition(gap: KnowledgeGap) -> BridgeRequest:
    """
    Dado um gap de conhecimento, produz um pedido concreto de 'ponte':
    - vetor ideal da ponte (semantic_query_vec)
    - especificação simbólica (BridgeSpec)
    - query textual (text_query)
    """
    bridge_vec = build_bridge_vector(gap.source_vec, gap.target_vec)
    spec = build_bridge_spec(gap)
    text_query = build_text_query(spec)

    # Filtros simples, você pode sofisticar depois
    filters = {
        "year_from": "2018",  # por ex., focar em coisas mais recentes
        "language": "en",     # ou "pt" se quiser restringir
    }

    return BridgeRequest(
        gap_id=gap.gap_id,
        semantic_query_vec=bridge_vec,
        bridge_spec=spec,
        text_query=text_query,
        filters=filters,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def compute_novelty_score(
    candidate_vec: np.ndarray,
    memory_vecs: List[np.ndarray],
    top_k: int = 20,
) -> float:
    """
    Novelty ~ 1 - max_sim_com_memória_existente (limitado ao top_k mais parecidos).
    Quanto mais distante de tudo, maior a novidade.
    """
    if not memory_vecs:
        return 1.0  # se não tem memória, tudo é novo

    sims = [cosine_sim(candidate_vec, m) for m in memory_vecs]
    sims_sorted = sorted(sims, reverse=True)[:top_k]
    max_sim = sims_sorted[0] if sims_sorted else 0.0
    novelty = 1.0 - max_sim
    return max(0.0, min(1.0, novelty))


def evaluate_bridge_impact(
    gap: KnowledgeGap,
    bridge_req: BridgeRequest,
    candidate_embedding: np.ndarray,
    candidate_meta: Dict,
    memory_vecs: List[np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> BridgeCandidate:
    """
    Calcula o impacto potencial de um candidato em fechar o gap.
    Score mistura:
      - similaridade com o vetor da ponte (bridge_vec)
      - similaridade com as extremidades (source/target)
      - novidade em relação à memória atual
    """

    sim_bridge = cosine_sim(candidate_embedding, bridge_req.semantic_query_vec)
    sim_source = cosine_sim(candidate_embedding, gap.source_vec)
    sim_target = cosine_sim(candidate_embedding, gap.target_vec)

    # Queremos que o candidato esteja razoavelmente próximo dos dois lados
    sim_source_target = min(sim_source, sim_target)

    novelty = compute_novelty_score(candidate_embedding, memory_vecs)

    final_score = (
        alpha * sim_bridge +
        beta * sim_source_target +
        gamma * novelty
    )

    title = candidate_meta.get("title", "Unknown")
    abstract = candidate_meta.get("abstract", "")
    doc_id = candidate_meta.get("doc_id", "")
    source = candidate_meta.get("source", "unknown")

    return BridgeCandidate(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        source=source,
        embedding=candidate_embedding,
        similarity_to_bridge=sim_bridge,
        similarity_to_source=sim_source,
        similarity_to_target=sim_target,
        novelty_score=novelty,
        final_score=final_score,
        metadata=candidate_meta,
    )


class BridgeAgent:
    """
    Agent responsible for bridging knowledge gaps.
    """
    def plan_bridge(self, gap: KnowledgeGap) -> BridgeRequest:
        return plan_bridge_acquisition(gap)

    def evaluate_candidate(
        self, 
        gap: KnowledgeGap, 
        request: BridgeRequest, 
        candidate_embedding: np.ndarray, 
        candidate_meta: Dict, 
        memory_vecs: List[np.ndarray]
    ) -> BridgeCandidate:
        return evaluate_bridge_impact(gap, request, candidate_embedding, candidate_meta, memory_vecs)
