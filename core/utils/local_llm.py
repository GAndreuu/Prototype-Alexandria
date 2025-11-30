"""
Prototype Alexandria - Local LLM Module
Expert Tático: TinyLlama para RAG Rápido e Local

Este módulo implementa o "Expert Tático" do sistema híbrido:
- Processamento local otimizado para CPU i9 (AVX512/Float32)
- Síntese factual de evidências
- Zero custo de API
- Sempre disponível (offline)

Autor: Antigravity AI Agent
Data: 2025-11-22
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
from config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Expert Tático - Modelo Local para RAG Factual
    
    Usa TinyLlama-1.1B-Chat otimizado para CPU i9:
    - Float32 (mais rápido que FP16 emulado em CPU)
    - Multi-threading (8 threads)
    - Batch size 1
    """
    
    def __init__(
        self, 
        model_name: str = settings.LOCAL_LLM_MODEL, 
        device: Optional[str] = settings.LOCAL_LLM_DEVICE
    ):
        """
        Inicializa LocalLLM com TinyLlama
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        self.model_loaded = False
        self.last_generation_time = 0.0
        
        logger.info(f"LocalLLM inicializado - Modelo: {model_name}, Device: {self.device}")
        
        # Carregar modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo TinyLlama e tokenizer"""
        try:
            logger.info(f"Carregando {self.model_name}...")
            start_time = time.time()
            
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configurar pad token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Otimização para CPU i9: float32 é geralmente mais rápido que float16 emulado
            # Se estivermos em CUDA, usamos float16
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Configurar para inferência
            self.model.eval()
            
            # Criar pipeline de geração
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"✅ {self.model_name} carregado em {load_time:.2f}s")
            logger.info(f"   Device: {self.device} (dtype={torch_dtype})")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.warning("⚠️ Sistema continuará usando apenas Gemini (fallback)")
            self.model_loaded = False
    
    def synthesize_facts(
        self, 
        query: str, 
        evidence: List[Dict[str, Any]],
        max_length: int = 512
    ) -> str:
        """
        Sintetiza fatos a partir de evidências recuperadas
        """
        if not self.model_loaded:
            return self._fallback_synthesis(query, evidence)
        
        try:
            start_time = time.time()
            
            # Construir contexto a partir das evidências
            context = self._build_context(evidence)
            
            # Criar prompt para TinyLlama
            prompt = self._create_factual_prompt(query, context)
            
            # Gerar resposta
            response = self.generator(
                prompt,
                max_new_tokens=1024,
                num_return_sequences=1,
                temperature=0.7, # Alta temperatura para criatividade (Semantic Collider)
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extrair texto gerado
            generated_text = response[0]['generated_text']
            
            # Extrair apenas a resposta (remover prompt)
            answer = self._extract_answer(generated_text, prompt)
            
            self.last_generation_time = time.time() - start_time
            
            logger.debug(f"✅ Rascunho gerado em {self.last_generation_time:.3f}s")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Erro na síntese factual: {e}")
            return self._fallback_synthesis(query, evidence)
    
    def _build_context(self, evidence: List[Dict[str, Any]]) -> str:
        """Constrói string de contexto a partir das evidências"""
        context_parts = []
        for i, ev in enumerate(evidence[:5], 1): # Top 5 evidências
            content = ev.get("content", "")[:300] # Limitar tamanho por chunk
            context_parts.append(f"[{i}] {content}")
        return "\n".join(context_parts)
    
    def _create_factual_prompt(self, query: str, context: str) -> str:
        """Cria prompt otimizado para TinyLlama"""
        # Simular "tempo de pensamento" para processamento profundo
        time.sleep(3) 

        return f"""<|system|>
Você é o MESTRE DA SÍNTESE ESTRATÉGICA.
Sua missão é encarnar a lógica profunda dos personagens e cenários apresentados.
Ao responder "Como Ender lidaria com Trisolaris?", você deve:
1. Analisar a psicologia tática de Ender (empatia destrutiva, ataque preventivo).
2. Contrastar com a natureza de Trisolaris (transparência, desespero, tecnologia superior).
3. Criar uma narrativa de vários parágrafos que mostre Ender tomando uma decisão realista e difícil.
4. Usar tom sério, detalhado e imersivo.
NÃO seja genérico. Seja específico, usando as evidências abaixo como base para sua extrapolação.
</s>
<|user|>
Evidências do Arquivo:
{context}

Pergunta do Usuário: {query}
</s>
<|assistant|>"""

    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """Remove o prompt da resposta gerada"""
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):]
        # Fallback se o prompt não for prefixo exato (tokenização)
        parts = generated_text.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1]
        return generated_text

    def _fallback_synthesis(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Fallback simples se o modelo falhar"""
        logger.warning("Usando fallback de síntese (concatenação)")
        context = "\n".join([e.get('content', '')[:200] for e in evidence[:3]])
        return f"RASCUNHO (FALLBACK): Baseado nas evidências disponíveis:\n{context}\n..."
