"""
Prototype Alexandria - Neural Oracle (Hybrid)
Implementação híbrida: TinyLlama local + Gemini cloud

Arquitetura "Córtex de Especialistas":
1. TinyLlama (Local): Expert Tático - RAG rápido e factual
2. Gemini (API): Expert Estratégico - Refinamento e crítica

Autor: Antigravity AI Agent
Data: 2025-11-22
"""

import os
import json
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.utils.local_llm import LocalLLM
from config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralOracle:
    """
    NeuralOracle - Processador Híbrido de Linguagem Natural
    
    Implementa arquitetura "Córtex de Especialistas":
    - TinyLlama: Síntese factual rápida (<100ms)
    - Gemini: Refinamento estratégico e crítica
    
    Capacidades:
    1. RAG Híbrido (Local + API)
    2. Síntese factual ultrarrápida
    3. Refinamento estilístico premium
    4. Fallback automático
    """
    
    def __init__(
        self, 
        model_name: str = settings.GEMINI_MODEL, 
        api_key: Optional[str] = None,
        use_hybrid: bool = settings.USE_HYBRID_MODE
    ):
        """
        Inicializa NeuralOracle Híbrido
        
        Args:
            model_name: Nome do modelo Gemini
            api_key: Chave API do Gemini
            use_hybrid: Se True, ativa pipeline híbrido (TinyLlama + Gemini)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_hybrid = use_hybrid
        
        # Inicializar Expert Tático (Local)
        self.local_llm = LocalLLM()
        
        # Inicializar Expert Estratégico (API)
        if not self.api_key:
            logger.warning("GEMINI_API_KEY não encontrada. NeuralOracle funcionará em modo LOCAL apenas.")
            self.gemini_model = None
            self.is_gemini_available = False
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel(model_name)
                self.is_gemini_available = True
                logger.info(f"Expert Estratégico (Gemini) inicializado: {model_name}")
            except Exception as e:
                logger.error(f"Erro ao inicializar Gemini: {e}")
                self.gemini_model = None
                self.is_gemini_available = False

    def synthesize(
        self, 
        query: str, 
        evidence: List[Dict[str, Any]], 
        context: Optional[str] = None,
        mode: str = "hybrid"  # "local", "hybrid", "gemini"
    ) -> str:
        """
        Síntese usando pipeline híbrido
        
        Args:
            query: Pergunta do usuário
            evidence: Evidências recuperadas
            context: Contexto adicional
            mode: Modo de operação ("local", "hybrid", "gemini")
            
        Returns:
            Resposta sintetizada
        """
        start_time = datetime.now()
        
        # 1. Modo LOCAL (Apenas TinyLlama)
        if mode == "local" or (mode == "hybrid" and not self.is_gemini_available):
            logger.info("🔍 Executando síntese LOCAL (TinyLlama)")
            return self.local_llm.synthesize_facts(query, evidence)
            
        # 2. Modo GEMINI (Apenas API)
        if mode == "gemini":
            logger.info("✨ Executando síntese GEMINI (API)")
            return self._gemini_synthesis(query, evidence, context)
            
        # 3. Modo HÍBRIDO (TinyLlama -> Gemini)
        if mode == "hybrid":
            logger.info("🚀 Executando pipeline HÍBRIDO")
            
            # Passo 1: Rascunho Factual (Local)
            draft = self.local_llm.synthesize_facts(query, evidence)
            
            # Passo 2: Refinamento Estratégico (API)
            refined = self._gemini_refine(draft, query, context)
            
            return refined
            
        return "Modo inválido"

    def _gemini_synthesis(self, query: str, evidence: List[Dict[str, Any]], context: Optional[str]) -> str:
        """Síntese direta via Gemini (lento, alta qualidade)"""
        if not self.is_gemini_available:
            return self.local_llm.synthesize_facts(query, evidence)
            
        prompt = self._build_causal_analysis_prompt(query, evidence, context)
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Erro no Gemini: {e}")
            return self.local_llm.synthesize_facts(query, evidence)

    def _gemini_refine(self, draft: str, query: str, context: Optional[str]) -> str:
        """Refina rascunho local usando Gemini"""
        if not self.is_gemini_available:
            return draft
            
        prompt = f"""Você é um editor especialista. Refine o seguinte rascunho factual para torná-lo mais fluente, profissional e bem estruturado. Mantenha os fatos, melhore o estilo.

PERGUNTA: {query}
CONTEXTO ADICIONAL: {context if context else 'N/A'}

RASCUNHO FACTUAL:
{draft}

RESPOSTA REFINADA:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Erro no refinamento Gemini: {e}")
            return draft

    def _build_causal_analysis_prompt(self, query: str, evidence: List[Dict[str, Any]], context: Optional[str] = None) -> str:
        """Constrói prompt para análise causal"""
        evidence_text = ""
        for i, ev in enumerate(evidence[:5]):
            content = ev.get('content', 'N/A')
            relevance = ev.get('relevance', 'N/A')
            evidence_text += f"Evidência {i+1} (relevância: {relevance}): {content}\n\n"
        
        return f"""Você é um Colisor Semântico (Semantic Collider) de uma Superinteligência Artificial.

CONTEXTO: {context if context else 'Síntese Avançada'}
CONSULTA: {query}

EVIDÊNCIAS:
{evidence_text}

INSTRUÇÕES DE COLISÃO E FUSÃO:
1. Não apenas resuma. **Colida** as evidências para gerar faíscas de novos insights.
2. **Funda** conceitos aparentemente desconexos encontrados no texto.
3. A resposta deve ser **longa, complexa e detalhada**.
4. Explore as implicações filosóficas, técnicas e causais.
5. Use uma linguagem sofisticada e abrangente.

Resposta da Fusão:"""

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "local_model": self.local_llm.model_name,
            "is_gemini_available": self.is_gemini_available,
            "is_local_available": self.local_llm.model_loaded,
            "mode": "hybrid" if self.use_hybrid else "single"
        }

def create_neural_oracle(api_key: Optional[str] = None, model_name: str = settings.GEMINI_MODEL) -> NeuralOracle:
    return NeuralOracle(model_name=model_name, api_key=api_key)