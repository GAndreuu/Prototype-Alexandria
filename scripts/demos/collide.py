"""
Semantic Collider
Colisor Semântico de Ideias.
Encontra conexões inesperadas entre uma fonte (ex: Ficção) e um alvo (ex: Papers Técnicos).

Uso:
    python scripts/collide.py --source "Nome da Fonte" --target "Nome do Alvo"
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Adicionar raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory.storage import LanceDBStorage
from config import settings

# Carregar variáveis de ambiente
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Collider")

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Tentar ler direto do arquivo se load_dotenv falhar (backup)
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                        api_key = line.strip().split("=")[1]
                        break
        except:
            pass
            
    if not api_key:
        logger.warning("⚠️ API KEY não encontrada (GOOGLE_API_KEY ou GEMINI_API_KEY). Hipóteses não serão geradas.")
        return None
        
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(settings.GEMINI_MODEL)

def collide(source_query: str, target_query: str):
    output_path = "collision_report.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        storage = LanceDBStorage()
        table = storage.table
        
        # 1. Buscar Chunks da Fonte
        source_filter = f"source LIKE '%{source_query}%'"
        source_results = table.search().where(source_filter).limit(10).to_list()
        
        if not source_results:
            log(f"❌ Nenhuma fonte encontrada com o termo: '{source_query}'")
            return

        log(f"⚡ Carregando colisor...")
        log(f"   - Fonte: '{source_query}' ({len(source_results)} chunks amostrados)")
        
        collisions = []
        
        log(f"   - Alvo: '{target_query}'")
        log("\n🔥 INICIANDO COLISÃO...\n")
        
        seen_pairs = set()

        for s_chunk in source_results:
            source_filename = Path(s_chunk['source']).name
            
            # Filtro: source não contém o nome do arquivo da fonte
            target_filter = f"source NOT LIKE '%{source_filename}%'"
            
            if target_query.lower() not in ["papers", "all"]:
                 target_filter += f" AND source LIKE '%{target_query}%'"
                 
            neighbors = table.search(s_chunk['vector']) \
                .where(target_filter) \
                .limit(5) \
                .to_list()
                
            for n in neighbors:
                target_filename = Path(n['source']).name
                
                if source_filename == target_filename:
                    continue
                    
                pair_id = f"{s_chunk['id']}-{n['id']}"
                if pair_id in seen_pairs:
                    continue
                seen_pairs.add(pair_id)

                dist = n['_distance']
                sim = 1 - (dist / 2) 
                
                if sim > 0.4: 
                    collisions.append({
                        'source_chunk': s_chunk,
                        'target_chunk': n,
                        'similarity': sim,
                        's_name': source_filename,
                        't_name': target_filename
                    })
        
        # Ordenar por similaridade
        collisions.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Dedup por conteúdo
        unique_collisions = []
        seen_content = set()
        for c in collisions:
            if c['t_name'] not in seen_content:
                unique_collisions.append(c)
                seen_content.add(c['t_name'])
            if len(unique_collisions) >= 3:
                break
                
        top_collisions = unique_collisions
        
        if not top_collisions:
            log("❄️ Nenhuma colisão significativa encontrada.")
            return

        # Exibir Resultados
        log(f"🔥 COLISÃO: \"{source_query}\" × {target_query}\n")
        log("Conexões encontradas:")
        
        prompt_context = ""
        
        for i, c in enumerate(top_collisions, 1):
            s_txt = c['source_chunk']['content']
            t_txt = c['target_chunk']['content']
            s_name = c['s_name']
            t_name = c['t_name']
            
            log(f"{i}. Chunk (de {s_name}):")
            log(f"\"{s_txt}\"")
            log(f"   ↔")
            log(f"   Chunk (de {t_name}):")
            log(f"\"{t_txt}\"")
            log(f"   Similaridade: {c['similarity']:.2f}\n")
            log("-" * 40 + "\n")
            
            prompt_context += f"Par {i}:\nFonte ({s_name}): {s_txt}\nAlvo ({t_name}): {t_txt}\n\n"

        # 3. Gerar Hipóteses com Gemini
        model = setup_gemini()
        if model:
            log("🧠 Gerando hipóteses com IA...\n")
            prompt = f"""
            Atue como um cientista visionário e interdisciplinar (ex: Carl Sagan misturado com Feynman).
            Analise as seguintes conexões semânticas encontradas entre uma obra de ficção (Contexto A) e literatura técnica/científica (Contexto B).
            
            {prompt_context}
            
            Gere 3 hipóteses científicas criativas e detalhadas baseadas nessas colisões.
            Para cada hipótese, use o seguinte formato estruturado:

            ### 1. [Nome da Hipótese Criativa]
            *   **Conexão Observada:** Explique como o conceito da ficção se liga ao conceito técnico.
            *   **Fundamentação Científica:** Aprofunde a ciência por trás do paper/conceito técnico.
            *   **Especulação Visionária:** Se a ficção fosse realidade, o que isso implicaria para a ciência?
            *   **Aplicação Prática:** Uma ideia de experimento ou tecnologia que poderia surgir dessa fusão.

            Seja profundo, técnico mas acessível, e não tenha medo de propor ideias ousadas.
            """
            
            try:
                response = model.generate_content(prompt)
                log("Hipóteses geradas:")
                log(response.text)
            except Exception as e:
                log(f"Erro ao gerar hipóteses: {e}")
                
    print(f"Relatório salvo em {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Collider")
    parser.add_argument("--source", required=True, help="Termo para buscar na fonte (ex: livro)")
    parser.add_argument("--target", default="papers", help="Termo para buscar no alvo (ex: papers)")
    
    args = parser.parse_args()
    collide(args.source, args.target)
