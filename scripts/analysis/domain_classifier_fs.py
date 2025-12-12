import sys
import os
from pathlib import Path
from collections import Counter
import re

OUTPUT_FILE = "domain_analysis.txt"

def log(msg):
    print(msg)
    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")

def classify_domains_fs():
    # Clear file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    log("🔍 Analisando domínios dos papers (via Sistema de Arquivos)...")
    
    try:
        library_path = Path("data/library/arxiv")
        if not library_path.exists():
            log(f"❌ Diretório não encontrado: {library_path}")
            return

        # Get all PDF files
        files = list(library_path.rglob("*.pdf"))
        log(f"📚 Total de arquivos na biblioteca: {len(files)}")
        
        # Domain Keywords Mapping
        domains = {
            "Physics & Quantum": ["physics", "quantum", "gravity", "magnetic", "spin", "particle", "thermodynamic", "optical", "laser", "energy", "matter", "galaxy", "cosmology", "astrophysics", "collider", "neutrino", "relativity", "nuclear"],
            "Computer Science & AI": ["learning", "neural", "network", "algorithm", "data", "computer", "vision", "robot", "intelligence", "cyber", "security", "graph", "optimization", "recognition", "detect", "adversarial", "deep", "agent", "reinforcement", "language", "model", "diffusion", "transformer", "generative"],
            "Mathematics": ["algebra", "geometry", "equation", "theorem", "proof", "stochastic", "manifold", "topology", "linear", "calculus", "differential", "logic", "discrete"],
            "Biology & Medicine": ["cell", "protein", "gene", "bio", "medical", "brain", "neuron", "disease", "cancer", "molecular", "drug", "genomic"],
            "Chemistry": ["chemical", "reaction", "synthesis", "molecular", "polymer", "carbon", "organic"],
            "Engineering": ["control", "system", "design", "mechanical", "electrical", "power", "vehicle", "sensor", "wireless"]
        }
        
        stats = Counter()
        unclassified = []
        
        log("🗂️ Classificando...")
        
        for f in files:
            filename = f.stem.lower()
            # Clean filename
            words = set(re.split(r'[_\-\s\.]', filename))
            
            classified = False
            for domain, keywords in domains.items():
                if any(k in words for k in keywords):
                    stats[domain] += 1
                    classified = True
                    break 
            
            if not classified:
                stats["Unclassified/Other"] += 1
                unclassified.append(filename)
                
        log("\n📊 Distribuição de Domínios (Baseada em Nomes de Arquivo):")
        log("=" * 60)
        total_classified = 0
        for domain, count in stats.most_common():
            percentage = (count / len(files)) * 100
            log(f"{domain:.<40} {count:>5} ({percentage:.1f}%)")
            total_classified += count
            
        log("=" * 60)
        
        if unclassified:
            log("\n❓ Top palavras em não classificados:")
            all_words = []
            for name in unclassified:
                all_words.extend([w for w in re.split(r'[_\-\s\.]', name) if len(w) > 3 and not w.isdigit()])
            
            common = Counter(all_words).most_common(20)
            for word, count in common:
                log(f"   - {word}: {count}")

    except Exception as e:
        log(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    classify_domains_fs()
