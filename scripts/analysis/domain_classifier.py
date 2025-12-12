import sys
import os
import pandas as pd
from pathlib import Path
from collections import Counter
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.storage import LanceDBStorage

def classify_domains():
    print("🔍 Analisando domínios dos papers...")
    
    try:
        storage = LanceDBStorage()
        df = storage.table.to_pandas()
        
        # Get unique sources
        sources = df['source'].unique()
        print(f"📚 Total de papers únicos processados: {len(sources)}")
        
        # Domain Keywords Mapping
        domains = {
            "Physics & Quantum": ["physics", "quantum", "gravity", "magnetic", "spin", "particle", "thermodynamic", "optical", "laser", "energy", "matter", "galaxy", "cosmology", "astrophysics", "collider", "neutrino"],
            "Computer Science & AI": ["learning", "neural", "network", "algorithm", "data", "computer", "vision", "robot", "intelligence", "cyber", "security", "graph", "optimization", "recognition", "detect", "adversarial"],
            "Mathematics": ["algebra", "geometry", "equation", "theorem", "proof", "stochastic", "manifold", "topology", "linear", "calculus", "differential"],
            "Biology & Medicine": ["cell", "protein", "gene", "bio", "medical", "brain", "neuron", "disease", "cancer", "molecular", "drug"],
            "Chemistry": ["chemical", "reaction", "synthesis", "molecular", "polymer", "carbon"],
            "Engineering": ["control", "system", "design", "mechanical", "electrical", "power", "vehicle"]
        }
        
        stats = Counter()
        unclassified = []
        
        print("🗂️ Classificando...")
        
        for source in sources:
            filename = Path(source).stem.lower()
            # Clean filename
            words = set(re.split(r'[_\-\s\.]', filename))
            
            classified = False
            for domain, keywords in domains.items():
                if any(k in words for k in keywords):
                    stats[domain] += 1
                    classified = True
                    # Don't break, a paper can belong to multiple domains (but for simple stats, maybe we count primary?)
                    # Let's count all matches to see overlap, or just first match for partition.
                    # Let's count first match for partition to sum up to total roughly.
                    break 
            
            if not classified:
                stats["Unclassified/Other"] += 1
                unclassified.append(filename)
                
        print("\n📊 Distribuição de Domínios:")
        print("=" * 40)
        total_classified = 0
        for domain, count in stats.most_common():
            percentage = (count / len(sources)) * 100
            print(f"{domain:.<30} {count:>5} ({percentage:.1f}%)")
            total_classified += count
            
        print("=" * 40)
        
        if unclassified:
            print("\n❓ Top palavras em não classificados:")
            all_words = []
            for name in unclassified:
                all_words.extend([w for w in re.split(r'[_\-\s\.]', name) if len(w) > 3])
            
            common = Counter(all_words).most_common(10)
            for word, count in common:
                print(f"   - {word}: {count}")

    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    classify_domains()
