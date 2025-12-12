#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-API Paper Ingestion para Alexandria
==========================================

APIs suportadas:
1. Semantic Scholar - PDFs + metadata (free API key recomendado)
2. CORE - Full-text open access (maior coleção)
3. OpenAlex - 100k req/dia (metadata rica)
4. ArXiv - Fallback (rate limited)

Autor: Alexandria Project
Data: 2025-12-08
"""

import os
import json
import time
import pathlib
import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import requests

# ==========================
# CONFIGURAÇÕES
# ==========================

# TUDO NA MESMA PASTA - data/library/arxiv
PDF_OUTPUT_DIR = pathlib.Path(r"c:\Users\G\Desktop\Alexandria\data\library\arxiv")
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ciclo perpétuo
PERPETUAL_MODE = True
MAX_PAPERS_PER_TOPIC = 100  # Aumentado para pegar mais
SLEEP_BETWEEN_TOPICS = 3    # Segundos entre tópicos
SLEEP_BETWEEN_CYCLES = 60   # Segundos entre ciclos completos

STATE_FILE = r"c:\Users\G\Desktop\Alexandria\data\multi_api_state.json"

# API Keys (opcional, aumenta rate limits)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
CORE_API_KEY = os.getenv("CORE_API_KEY", "")

USER_AGENT = "Alexandria-Agent/2.0 (research-project; contact@alexandria.local)"

# Rate limits por API
RATE_LIMITS = {
    "semantic_scholar": 100,  # 100 req/5min sem key, 1 req/seg com key
    "core": 10,               # 5 req/10seg sem registro
    "openalex": 100000,       # 100k/dia
    "arxiv": 3                # muito restrito
}

# ==========================
# BASE CLASS
# ==========================

@dataclass
class Paper:
    """Estrutura de paper normalizada"""
    title: str
    abstract: str
    authors: List[str]
    year: int
    pdf_url: Optional[str]
    doi: Optional[str]
    source: str  # qual API
    metadata: Dict

class PaperAPI(ABC):
    """Interface para APIs de papers"""
    
    @abstractmethod
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        pass
    
    @abstractmethod
    def get_rate_limit_wait(self) -> float:
        pass

# ==========================
# SEMANTIC SCHOLAR API
# ==========================

class SemanticScholarAPI(PaperAPI):
    """
    Semantic Scholar API
    - 100 req/5min sem key
    - 1 req/seg com key (grátis, precisa registrar)
    - Excelente para PDFs de acesso aberto
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.headers = {"User-Agent": USER_AGENT}
        if api_key:
            self.headers["x-api-key"] = api_key
    
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        """Busca papers no Semantic Scholar"""
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,openAccessPdf,externalIds,citationCount"
        }
        
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if resp.status_code == 429:
                print(f"  [S2] Rate limit. Esperando 60s...")
                time.sleep(60)
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("data", []):
                pdf_url = None
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url")
                
                paper = Paper(
                    title=item.get("title", ""),
                    abstract=item.get("abstract", "") or "",
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=item.get("year") or 0,
                    pdf_url=pdf_url,
                    doi=item.get("externalIds", {}).get("DOI"),
                    source="semantic_scholar",
                    metadata={"citations": item.get("citationCount", 0)}
                )
                papers.append(paper)
            
            print(f"  [S2] Encontrados {len(papers)} papers, {sum(1 for p in papers if p.pdf_url)} com PDF")
            return papers
            
        except Exception as e:
            print(f"  [S2] Erro: {e}")
            return []
    
    def get_rate_limit_wait(self) -> float:
        return 1.0 if self.api_key else 3.0

# ==========================
# CORE API
# ==========================

class COREAPI(PaperAPI):
    """
    CORE API - Maior coleção de open access
    - Full-text disponível
    - Rate limit: 5 req/10seg sem registro
    - Mais generoso que ArXiv
    """
    
    BASE_URL = "https://api.core.ac.uk/v3"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.headers = {"User-Agent": USER_AGENT}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        """Busca papers no CORE"""
        url = f"{self.BASE_URL}/search/works"
        params = {
            "q": query,
            "limit": min(limit, 100),
        }
        
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if resp.status_code == 429:
                print(f"  [CORE] Rate limit. Esperando 30s...")
                time.sleep(30)
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("results", []):
                pdf_url = item.get("downloadUrl") or item.get("sourceFulltextUrls", [None])[0] if item.get("sourceFulltextUrls") else None
                
                paper = Paper(
                    title=item.get("title", ""),
                    abstract=item.get("abstract", "") or "",
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=item.get("yearPublished") or 0,
                    pdf_url=pdf_url,
                    doi=item.get("doi"),
                    source="core",
                    metadata={"language": item.get("language", "")}
                )
                papers.append(paper)
            
            print(f"  [CORE] Encontrados {len(papers)} papers, {sum(1 for p in papers if p.pdf_url)} com PDF")
            return papers
            
        except Exception as e:
            print(f"  [CORE] Erro: {e}")
            return []
    
    def get_rate_limit_wait(self) -> float:
        return 2.0 if self.api_key else 3.0

# ==========================
# OPENALEX API
# ==========================

class OpenAlexAPI(PaperAPI):
    """
    OpenAlex API - 100k req/dia
    - Metadata muito rica
    - PDFs via links externos
    - Sem autenticação necessária
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self):
        self.headers = {"User-Agent": USER_AGENT}
    
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        """Busca papers no OpenAlex"""
        url = f"{self.BASE_URL}/works"
        params = {
            "search": query,
            "per_page": min(limit, 200),
            "filter": "is_oa:true",  # Apenas open access
        }
        
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("results", []):
                # Encontrar PDF URL
                pdf_url = None
                if item.get("open_access", {}).get("oa_url"):
                    pdf_url = item["open_access"]["oa_url"]
                
                # Extrair ano
                year = 0
                if item.get("publication_year"):
                    year = item["publication_year"]
                
                paper = Paper(
                    title=item.get("title", ""),
                    abstract=item.get("abstract", "") or "",
                    authors=[a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])],
                    year=year,
                    pdf_url=pdf_url,
                    doi=item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None,
                    source="openalex",
                    metadata={"cited_by_count": item.get("cited_by_count", 0)}
                )
                papers.append(paper)
            
            print(f"  [OpenAlex] Encontrados {len(papers)} papers, {sum(1 for p in papers if p.pdf_url)} com PDF")
            return papers
            
        except Exception as e:
            print(f"  [OpenAlex] Erro: {e}")
            return []
    
    def get_rate_limit_wait(self) -> float:
        return 0.1  # Muito generoso

# ==========================
# MULTI-API ORCHESTRATOR
# ==========================

class MultiAPIOrchestrator:
    """
    Orquestra múltiplas APIs para maximizar coleta de papers
    """
    
    def __init__(self):
        self.apis = {
            "semantic_scholar": SemanticScholarAPI(SEMANTIC_SCHOLAR_API_KEY),
            "core": COREAPI(CORE_API_KEY),
            "openalex": OpenAlexAPI(),
        }
        self.downloaded_titles = set()
        self._load_existing_files()
    
    def _load_existing_files(self):
        """Carrega títulos já baixados"""
        if PDF_OUTPUT_DIR.exists():
            for f in PDF_OUTPUT_DIR.glob("*.pdf"):
                norm_title = self._normalize_title(f.stem)
                self.downloaded_titles.add(norm_title)
        print(f"[Orchestrator] {len(self.downloaded_titles)} papers já na biblioteca")
    
    def _normalize_title(self, title: str) -> str:
        """Normaliza título para comparação"""
        return re.sub(r'[^a-z0-9]', '', title.lower())
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitiza título para nome de arquivo"""
        safe = re.sub(r'[\\/*?:"<>|]', '_', title)
        safe = " ".join(safe.split())
        return safe[:150].strip()
    
    def search_all_apis(self, query: str, limit_per_api: int = 30) -> List[Paper]:
        """Busca em todas as APIs e deduplica"""
        all_papers = []
        seen_titles = set()
        
        for api_name, api in self.apis.items():
            print(f"\n[{api_name.upper()}] Buscando: '{query[:50]}...'")
            
            try:
                papers = api.search(query, limit_per_api)
                
                for paper in papers:
                    norm_title = self._normalize_title(paper.title)
                    if norm_title not in seen_titles and norm_title not in self.downloaded_titles:
                        seen_titles.add(norm_title)
                        all_papers.append(paper)
                
                # Respeitar rate limit
                wait_time = api.get_rate_limit_wait()
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"[{api_name}] Erro: {e}")
        
        print(f"\n[TOTAL] {len(all_papers)} papers únicos encontrados")
        return all_papers
    
    def download_paper(self, paper: Paper) -> bool:
        """Baixa PDF de um paper"""
        if not paper.pdf_url:
            return False
        
        filename = self._sanitize_filename(paper.title)
        if not filename:
            return False
        
        filepath = PDF_OUTPUT_DIR / f"{filename}.pdf"
        
        if filepath.exists():
            return False
        
        try:
            print(f"  [DOWNLOAD] {filename[:60]}...")
            resp = requests.get(
                paper.pdf_url, 
                headers={"User-Agent": USER_AGENT},
                timeout=60,
                allow_redirects=True
            )
            
            if resp.status_code == 200 and len(resp.content) > 1000:
                filepath.write_bytes(resp.content)
                self.downloaded_titles.add(self._normalize_title(paper.title))
                print(f"           ✓ Sucesso ({len(resp.content)//1024}KB)")
                time.sleep(random.uniform(2, 5))
                return True
            else:
                print(f"           ✗ Falha (status={resp.status_code})")
                return False
                
        except Exception as e:
            print(f"           ✗ Erro: {e}")
            return False
    
    def run_collection(self, topics: Dict[str, str], papers_per_topic: int = 30):
        """Executa coleta para todos os tópicos"""
        total_downloaded = 0
        
        for topic_name, query in topics.items():
            print(f"\n{'='*60}")
            print(f"TÓPICO: {topic_name}")
            print(f"{'='*60}")
            
            # Simplificar query para APIs que não suportam sintaxe ArXiv
            simple_query = self._simplify_query(query)
            
            papers = self.search_all_apis(simple_query, papers_per_topic)
            
            # Priorizar papers com PDF
            papers_with_pdf = [p for p in papers if p.pdf_url]
            
            print(f"\n[DOWNLOAD] Baixando {len(papers_with_pdf)} PDFs...")
            
            for paper in papers_with_pdf[:papers_per_topic]:
                if self.download_paper(paper):
                    total_downloaded += 1
            
            print(f"\n[TÓPICO] Concluído. Total baixados até agora: {total_downloaded}")
            
            # Pausa entre tópicos
            time.sleep(5)
        
        print(f"\n{'='*60}")
        print(f"COLETA FINALIZADA: {total_downloaded} papers baixados")
        print(f"{'='*60}")
    
    def _simplify_query(self, arxiv_query: str) -> str:
        """Converte query ArXiv para query simples"""
        # Remove sintaxe ArXiv
        simple = arxiv_query.replace("all:", "").replace("'", "")
        simple = re.sub(r'[()]', ' ', simple)
        simple = re.sub(r'\s+AND\s+', ' ', simple)
        simple = re.sub(r'\s+OR\s+', ' ', simple)
        simple = re.sub(r'\s+', ' ', simple).strip()
        
        # Pegar os termos mais importantes (primeiros 5)
        terms = [t.strip('"') for t in simple.split() if len(t) > 3]
        return ' '.join(terms[:5])


# ==========================
# TÓPICOS DE META-LEARNING
# ==========================

# Importar tópicos se disponível
try:
    from meta_learning_topics import META_LEARNING_TOPICS
    TOPICS = META_LEARNING_TOPICS
except ImportError:
    print("[WARN] meta_learning_topics.py não encontrado, usando tópicos básicos")
    TOPICS = {
        "meta_learning": "meta-learning MAML few-shot learning",
        "metacognition": "metacognition self-monitoring neural network",
        "free_energy": "free energy principle Friston predictive coding",
        "hebbian": "Hebbian learning plasticity STDP",
        "information_geometry": "information geometry Fisher natural gradient",
    }


# ==========================
# MAIN - CICLO PERPÉTUO
# ==========================

def main():
    print("="*60)
    print("🚀 ALEXANDRIA - Multi-API Paper Collection")
    print("   MODO PERPÉTUO - VAI RODAR A NOITE TODA!")
    print("="*60)
    print(f"Output: {PDF_OUTPUT_DIR}")
    print(f"Tópicos: {len(TOPICS)}")
    print(f"Papers por tópico: {MAX_PAPERS_PER_TOPIC}")
    print(f"Modo perpétuo: {PERPETUAL_MODE}")
    print()
    
    orchestrator = MultiAPIOrchestrator()
    
    cycle = 0
    total_session = 0
    
    while True:
        cycle += 1
        print(f"\n{'#'*60}")
        print(f"### CICLO {cycle} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"### Papers baixados nesta sessão: {total_session}")
        print(f"### Papers na biblioteca: {len(orchestrator.downloaded_titles)}")
        print(f"{'#'*60}")
        
        for topic_name, query in TOPICS.items():
            print(f"\n{'='*60}")
            print(f"TÓPICO: {topic_name}")
            print(f"{'='*60}")
            
            # Simplificar query
            simple_query = orchestrator._simplify_query(query)
            
            # Buscar em todas as APIs
            papers = orchestrator.search_all_apis(simple_query, MAX_PAPERS_PER_TOPIC)
            
            # Baixar papers com PDF
            papers_with_pdf = [p for p in papers if p.pdf_url]
            
            if papers_with_pdf:
                print(f"\n[DOWNLOAD] Baixando {len(papers_with_pdf)} PDFs...")
                
                for paper in papers_with_pdf:
                    if orchestrator.download_paper(paper):
                        total_session += 1
            else:
                print(f"[SKIP] Nenhum paper novo com PDF")
            
            # Pausa curta entre tópicos
            time.sleep(SLEEP_BETWEEN_TOPICS)
        
        print(f"\n{'#'*60}")
        print(f"### CICLO {cycle} COMPLETO!")
        print(f"### Total baixados nesta sessão: {total_session}")
        print(f"### Total na biblioteca: {len(orchestrator.downloaded_titles)}")
        print(f"{'#'*60}")
        
        if not PERPETUAL_MODE:
            print("\nModo perpétuo desativado. Finalizando.")
            break
        
        # Recarregar arquivos existentes (caso outro processo tenha adicionado)
        orchestrator._load_existing_files()
        
        print(f"\n⏳ Aguardando {SLEEP_BETWEEN_CYCLES}s antes do próximo ciclo...")
        print(f"   Próximo ciclo às: {time.strftime('%H:%M:%S', time.localtime(time.time() + SLEEP_BETWEEN_CYCLES))}")
        time.sleep(SLEEP_BETWEEN_CYCLES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrompido pelo usuário. Tchau!")

