#!/usr/bin/env python3
"""
Context Mapper V1 - Implementação do conceito 'Cosmic Garden'
Gera um mapa topológico e um índice invertido de intenção para o projeto.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Configuração
ROOT_DIR = Path(".")
OUTPUT_FILE = Path(".agent/context_map.json")
IGNORE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.idea', '.vscode', 'node_modules', 'site-packages', 'data', 'library', 'lancedb_store', 'logs', 'reports'}
IGNORE_EXTS = {'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot', '.zip', '.tar', '.gz'}

def normalize_and_tokenize(name: str) -> List[str]:
    """
    Transforma nomes de arquivos (CamelCase, snake_case) em tokens de busca.
    Ex: 'UserAuthentication.py' -> ['user', 'authentication', 'py']
    """
    # Remove extensão para análise (mas mantém como token se útil)
    stem = name
    if '.' in name:
        stem = name.rsplit('.', 1)[0]
    
    # Separa CamelCase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', stem)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    
    # Separa símbolos
    tokens = re.split(r'[^a-zA-Z0-9]', s2)
    
    # Adiciona a extensão como token também
    if '.' in name:
        tokens.append(name.rsplit('.', 1)[1])
        
    return [t.lower() for t in tokens if t]

def scan_topology(root: Path) -> List[Dict[str, Any]]:
    """
    Varre o diretório recursivamente gerando uma árvore topológica plana.
    Focamos em uma lista plana de "nós" para facilitar o índice invertido.
    """
    topology = []
    
    for path in root.rglob('*'):
        # Filtros de exclusão
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix in IGNORE_EXTS:
            continue
        if not path.is_file(): # Focamos em arquivos para o contexto
            continue
            
        rel_path = path.relative_to(root)
        
        node = {
            "path": str(rel_path),
            "name": path.name,
            "type": "file",
            "size": path.stat().st_size,
            "tokens": normalize_and_tokenize(path.name)
        }
        
        # Adiciona tokens dos diretórios pais para contexto
        for part in rel_path.parent.parts:
            node["tokens"].extend(normalize_and_tokenize(part))
            
        # Remove duplicatas de tokens
        node["tokens"] = list(set(node["tokens"]))
        
        topology.append(node)
        
    return topology

def build_inverted_index(topology: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Cria o mapa intenção -> arquivos.
    """
    index = defaultdict(list)
    
    for node in topology:
        for token in node["tokens"]:
            # Evita duplicatas de caminho na mesma chave
            if node["path"] not in index[token]:
                index[token].append(node["path"])
                
    return dict(index)

def main():
    print(f"🌌 Cosmic Garden: Context Mapper Iniciado")
    print(f"📂 Raiz: {ROOT_DIR.absolute()}")
    
    # 1. Escanear
    print("running scan...")
    topology = scan_topology(ROOT_DIR)
    print(f"✓ Encontrados {len(topology)} arquivos relevantes.")
    
    # 2. Indexar
    print("building index...")
    inverted_index = build_inverted_index(topology)
    print(f"✓ Indexados {len(inverted_index)} termos únicos.")
    
    # 3. Salvar
    data = {
        "metadata": {
            "generated_at": "runtime",
            "total_files": len(topology),
            "total_terms": len(inverted_index)
        },
        "topology": topology,
        "inverted_index": inverted_index
    }
    
    # Garantir que diretório .agent existe
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Mapa salvo em: {OUTPUT_FILE}")
    print("Conclusion: 'A gravidade topológica foi estabelecida.'")

if __name__ == "__main__":
    main()
