#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para coletar papers massivamente da API do arXiv em ciclos de 50 por macro-tópico.
Adaptado para o ambiente Alexandria com verificação ROBUSTA de duplicatas.

- Usa: http://export.arxiv.org/api/query
- Salva PDFs com o NOME DO TÍTULO.
- Salva na pasta 'data/library/arxiv'.
- Verifica duplicatas por:
    1. Nome exato do arquivo.
    2. Título normalizado (ignora case, espaços e símbolos).

Requisitos:
    pip install feedparser requests
"""

import os
import json
import time
import pathlib
import urllib.parse as up
import random
import re

import feedparser
import requests

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================

BASE_URL = "http://export.arxiv.org/api/query"
# Caminho absoluto para state file
STATE_FILE = r"c:\Users\G\Desktop\Alexandria\data\arxiv_state.json"

# Batch size
BATCH_SIZE = 50

# Tempo entre requisições da API (busca)
SLEEP_SECONDS_BETWEEN_REQUESTS = 10

# Ativado Download
DOWNLOAD_PDFS = True

# Caminho absoluto para output (agora corrigido para a pasta principal)
PDF_OUTPUT_DIR = pathlib.Path(r"c:\Users\G\Desktop\Alexandria\data\library\arxiv")
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Alexandria-Agent/1.0 (research-project; mailto:admin@alexandria.local)"

# Estado global dos arquivos existentes (set de strings normalizadas)
EXISTING_FILES_NORMALIZED = set()


# ==========================
# MACRO-TÓPICOS (Original + Meta-Learning)
# ==========================

# Importar tópicos de meta-learning
try:
    from meta_learning_topics import META_LEARNING_TOPICS
    HAS_META_TOPICS = True
except ImportError:
    META_LEARNING_TOPICS = {}
    HAS_META_TOPICS = False

# Tópicos originais do projeto Alexandria
ORIGINAL_TOPICS = {
    "neural_dynamics_and_control": (
        '('
        '(all:"spiking neural network" OR all:"spiking neuron" OR all:neuromorphic) '
        'OR '
        '((all:"control barrier function" OR all:"control barrier certificate") '
        ' AND (all:"neural network" OR all:"learning-based control"))'
        ')'
    ),
    "geometric_and_physical_ml": (
        '('
        '(all:"geometric deep learning" OR all:"manifold learning" OR all:"Riemannian") '
        'OR '
        '((all:"finite-strain" OR all:"finite strain") '
        ' AND (all:"microelasticity" OR all:"micro-elasticity" OR all:"microstructure"))'
        ')'
    ),
    "planning_and_cps": (
        '('
        '((all:"AI planning" OR all:"automated planning") '
        '  AND (all:"cyber-physical system" OR all:CPS)) '
        'OR '
        '((all:"implicit planning" OR all:"planning ability") '
        '  AND (all:"language model" OR all:"LLM"))'
        ')'
    ),
    "llm_instruction_kr_reasoning": (
        '('
        '(all:"in-context learning" OR all:"instruction following" OR all:"instruction tuning" '
        ' OR all:"chain-of-thought prompting" OR all:"rule learning" OR all:"symbolic rules" '
        ' OR all:"knowledge representation" OR all:"knowledge graph" OR all:ontology '
        ' OR all:"neuro-symbolic" OR all:"neurosymbolic") '
        'AND '
        '(all:"language model" OR all:"large language model" OR all:"LLM agent" OR all:"LLM")'
        ')'
    ),
    "vision_reasoning_and_iqa": (
        '('
        '((all:"image quality assessment" OR all:IQA) '
        '  AND (all:"causal" OR all:"counterfactual" OR all:"abductive reasoning")) '
        'OR '
        '((all:"visual puzzle" OR all:"puzzle-based visual" OR all:"visual abstract reasoning") '
        '  AND (all:"chain-of-thought" OR all:"step-by-step reasoning" OR all:"error detection")) '
        'OR '
        '((all:"probabilistic abduction" OR all:"abductive reasoning") '
        '  AND (all:"vector symbolic architecture" OR all:"Holographic Reduced Representation" OR all:VSA)) '
        'OR '
        '((all:"3D object detection" AND all:"multi-view") '
        '  OR (all:"BEV" AND all:"multi-camera"))'
        ')'
    ),
    "medical_and_domain_reasoning": (
        '('
        '(all:"medical dialogue" OR all:"clinical conversation" OR all:"doctor-patient dialogue") '
        'AND '
        '(all:"diagnostic reasoning" OR all:"clinical reasoning")'
        ')'
    ),
    "sparse_and_discrete_representations": (
        '('
        '((all:"sparse dictionary learning" OR all:"dictionary learning" OR all:"sparse coding" '
        '   OR all:"compressive sensing" OR all:"compressed sensing") '
        '  AND '
        ' (all:"image reconstruction" OR all:"inverse problem" OR all:"sparse" OR all:"Tikhonov")) '
        'OR '
        '(all:"VQ-VAE" OR all:"vector quantized variational autoencoder" OR all:"discrete representation learning")'
        ')'
    ),
    "epistemic_logic_and_rationality": (
        '('
        '(all:"epistemic logic" OR all:"common knowledge" OR all:"knowledge and belief") '
        'AND '
        '(all:"game theory" OR all:"rationality")'
        ')'
    ),
    "astro_and_dark_matter": (
        '('
        '(all:"dark matter halo" OR all:"galactic halo" OR all:"Dehnen profile") '
        'AND '
        '(all:"black hole" OR all:"gravitational lensing")'
        ')'
    ),
}

# Combinar tópicos: Original (9) + Meta-Learning (36) = 45 tópicos
TOPICS = {**ORIGINAL_TOPICS, **META_LEARNING_TOPICS}

print(f"[INFO] Tópicos carregados: {len(ORIGINAL_TOPICS)} originais + {len(META_LEARNING_TOPICS)} meta-learning = {len(TOPICS)} total")

# ==========================
# UTILITÁRIOS
# ==========================

def normalize_string(s):
    """Normaliza string para comparação: lowercase, sem acentos, sem símbolos."""
    # Remove extensão .pdf se houver no final
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    # Remove símbolos não alfanuméricos
    return re.sub(r'[^a-z0-9]', '', s.lower())

def sanitize_filename(title):
    """Remove caracteres inválidos e limita tamanho."""
    safe = re.sub(r'[\\/*?:"<>|]', '_', title)
    safe = "".join(ch for ch in safe if ch.isprintable())
    safe = " ".join(safe.split())
    if len(safe) > 150:
        safe = safe[:150]
    return safe.strip()

def build_existing_index(directory: pathlib.Path):
    """Monta índice de arquivos normalizados."""
    print("--- INDEXANDO BIBLIOTECA EXISTENTE ---")
    count = 0
    if directory.exists():
        for f in directory.glob("*.pdf"):
            norm_name = normalize_string(f.name)
            EXISTING_FILES_NORMALIZED.add(norm_name)
            count += 1
    print(f"Indexados {count} arquivos para prevenção de duplicatas.\n")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {topic: {"start": 0} for topic in TOPICS.keys()}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
         state = {}
    for topic in TOPICS.keys():
        state.setdefault(topic, {"start": 0})
    return state

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERRO] Ao salvar estado: {e}")

# ==========================
# CHAMADA À API
# ==========================

def query_arxiv(search_query: str, start: int, max_results: int = BATCH_SIZE):
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}

    print(f"\n[ARXIV] BUSCA: start={start}, max={max_results}, query='{search_query[:30]}...'")
    
    for attempt in range(3):
        try:
            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = (attempt + 1) * 15
                print(f"  [429] Limit. Esperando {wait}s...")
                time.sleep(wait)
            else:
                resp.raise_for_status()
        except Exception as e:
            print(f"  [ERRO] Tentativa {attempt+1} falhou: {e}")
            time.sleep(5)
    else:
        print("  [ERRO] Falha ao buscar feed.")
        return None, 0, 0

    feed = feedparser.parse(resp.text)
    try:
        total_results = int(feed.feed.opensearch_totalresults)
    except Exception:
        total_results = 0
    try:
        items_per_page = int(feed.feed.opensearch_itemsperpage)
    except Exception:
        items_per_page = len(feed.entries)

    return feed, total_results, items_per_page

# ==========================
# DOWNLOAD PDF
# ==========================

def download_pdf_for_entry(entry, output_dir: pathlib.Path):
    pdf_url = None
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            pdf_url = link.get("href")
            break

    if not pdf_url:
        print("  [WARN] Link PDF não encontrado.")
        return

    # Nome base
    title = entry.title.replace("\n", " ").strip()
    safe_title = sanitize_filename(title)
    if not safe_title:
        safe_title = entry.id.split("/abs/")[-1]

    filename = output_dir / f"{safe_title}.pdf"

    # Verificação Dupla:
    # 1. Arquivo exato existe?
    if filename.exists():
        print(f"  [SKIP] '{filename.name}' já existe.")
        return

    # 2. Título normalizado existe em qualquer outro arquivo?
    norm_title = normalize_string(title)
    if norm_title in EXISTING_FILES_NORMALIZED:
        print(f"  [SKIP] '{filename.name}' (Título já existe na biblioteca).")
        return

    print(f"  [PDF] Baixando: '{filename.name}'...")
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(3):
        try:
            resp = requests.get(pdf_url, headers=headers, timeout=60)
            if resp.status_code == 200:
                filename.write_bytes(resp.content)
                print("        Sucesso.")
                
                # Adiciona ao índice em memória para evitar dupes no mesmo processo
                EXISTING_FILES_NORMALIZED.add(norm_title)
                
                time.sleep(random.uniform(5, 10)) 
                return
            elif resp.status_code == 429:
                wait = (attempt + 1) * 30
                print(f"        [429] Rate Limit PDF. Esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"        Erro HTTP: {resp.status_code}")
                return
        except Exception as e:
             print(f"        Erro download: {e}")
             time.sleep(5)

# ==========================
# BATCH + LOOP
# ==========================

def fetch_batch_for_topic(topic_key: str, state: dict, batch_size: int = BATCH_SIZE):
    search_query = TOPICS[topic_key]
    start = state[topic_key].get("start", 0)

    feed, total_results, items_per_page = query_arxiv(search_query, start, max_results=batch_size)
    if feed is None: return []

    entries = feed.entries

    if not entries:
        print(f"[INFO] Sem resultados para '{topic_key}'. Reiniciando.")
        start = 0
        feed, total_results, items_per_page = query_arxiv(search_query, start, max_results=batch_size)
        if feed: entries = feed.entries
        if not entries: return []

    next_start = start + batch_size
    if total_results > 0 and next_start >= total_results:
        print(f"[INFO] Tópico '{topic_key}' completou o ciclo. Reiniciando.")
        next_start = 0

    state[topic_key]["start"] = next_start
    save_state(state)
    print(f"[INFO] Atualizado start para {next_start} (Total aprox: {total_results})")
    return entries

def process_entries(topic_key: str, entries):
    print(f"\n===== TÓPICO: {topic_key} | QTD: {len(entries)} =====")
    for i, e in enumerate(entries, start=1):
        if DOWNLOAD_PDFS:
            try:
                download_pdf_for_entry(e, PDF_OUTPUT_DIR)
            except Exception as ex:
                print(f"  [ERRO FATAL PDF] {ex}")

def main():
    print("=== INICIANDO COLETA MASSIVA DO ARXIV (VERIFICAÇÃO DE DUPLICATAS) ===")
    print(f"Output Dir: {PDF_OUTPUT_DIR}")
    
    # Constrói índice inicial
    build_existing_index(PDF_OUTPUT_DIR)
    
    state = load_state()

    cycle = 0
    MAX_CYCLES = 5 
    
    while cycle < MAX_CYCLES:
        cycle += 1
        print(f"\n\n>>> CICLO GLOBAL {cycle}/{MAX_CYCLES} <<<")
        
        for topic_key in TOPICS.keys():
            print(f"\n--- Processando Tópico: {topic_key} ---")
            entries = fetch_batch_for_topic(topic_key, state, batch_size=BATCH_SIZE)
            if entries:
                process_entries(topic_key, entries)

            print(f"Dormindo {SLEEP_SECONDS_BETWEEN_REQUESTS}s entre tópicos...")
            time.sleep(SLEEP_SECONDS_BETWEEN_REQUESTS)
    print("\n=== FIM DA COLETA ===")

if __name__ == "__main__":
    main()
