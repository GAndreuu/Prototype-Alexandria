import json
import random

with open('data/validation_results.json', 'r') as f:
    data = json.load(f)

connections = data['alexandria']['connections']

print('='*70)
print('10 CONEXÕES PARA VALIDAÇÃO MANUAL')
print('='*70)

random.seed(42)
sample = random.sample(connections, min(15, len(connections)))

for i, c in enumerate(sample[:10], 1):
    paper_a = c['paper_a'].replace('data\\library\\arxiv\\', '').replace('data/library/arxiv/', '')
    paper_b = c['paper_b'].replace('data\\library\\arxiv\\', '').replace('data/library/arxiv/', '')
    method = c.get('method', '?')
    
    print(f"\n[{i}] PAPER A: {paper_a}")
    print(f"    PAPER B: {paper_b}")
    print(f"    Method: {method}")
    
    if 'shared_codes' in c:
        print(f"    Shared VQ codes: {c['shared_codes']}")
    if 'embedding_distance' in c:
        print(f"    Embedding distance: {c['embedding_distance']:.2f}")
    if 'mycelial_weight' in c:
        print(f"    Mycelial weight: {c['mycelial_weight']:.2f}")
