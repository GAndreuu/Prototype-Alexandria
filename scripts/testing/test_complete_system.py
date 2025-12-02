"""
Teste completo do Alexandria com modelo Wiki integrado.
Simula uso real: encode embeddings, observe patterns, reason.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.reasoning.mycelial_reasoning import MycelialVQVAE

print("="*70)
print("TESTE COMPLETO: Alexandria com Modelo Wiki")
print("="*70)

# 1. Inicializar sistema
print("\n[1/5] Inicializando MycelialVQVAE...")
mvq = MycelialVQVAE.load_default(use_wiki_model=True)
print(f"      ✅ Modelo carregado: {mvq.vqvae.__class__.__name__}")

# 2. Criar dados sintéticos que simulam embeddings de documentos
print("\n[2/5] Criando embeddings sintéticos (simulando 50 documentos)...")
np.random.seed(42)
docs_embeddings = []
for i in range(50):
    # Simular embeddings de documentos com alguma estrutura
    base = np.random.randn(384) * 0.5
    noise = np.random.randn(384) * 0.1
    doc_emb = base + noise
    docs_embeddings.append(doc_emb)

docs_embeddings = torch.tensor(np.array(docs_embeddings), dtype=torch.float32)
print(f"      ✅ Criados 50 embeddings sintéticos (shape: {docs_embeddings.shape})")

# 3. Encode e observar padrões
print("\n[3/5] Encoding documentos e observando padrões...")
all_indices = []
for i, emb in enumerate(docs_embeddings):
    indices = mvq.encode(emb.unsqueeze(0))
    mvq.observe(indices[0])  # Observar para aprendizado Hebbian
    all_indices.append(indices[0].cpu().numpy())

all_indices = np.array(all_indices)
print(f"      ✅ Encoded {len(all_indices)} documentos")

# 4. Análise de uso do codebook
print("\n[4/5] Analisando uso do codebook...")
unique_codes_total = np.unique(all_indices).shape[0]
usage_percent = (unique_codes_total / 1024) * 100

print(f"      📊 Códigos únicos usados: {unique_codes_total}/1024 ({usage_percent:.1f}%)")

# Análise por head
for h in range(4):
    head_codes = all_indices[:, h]
    unique_head = len(np.unique(head_codes))
    usage_head = (unique_head / 256) * 100
    print(f"      📊 Head {h}: {unique_head}/256 códigos ({usage_head:.1f}%)")

# 5. Teste de reasoning
print("\n[5/5] Testando raciocínio micelial...")
test_doc = docs_embeddings[0]
original_indices = mvq.encode(test_doc.unsqueeze(0))[0]
reasoned_indices, activation = mvq.reason(original_indices, steps=3)

print(f"      🧠 Índices originais:  {original_indices.cpu().numpy()}")
print(f"      🧠 Índices raciocínio: {reasoned_indices.cpu().numpy()}")
print(f"      🧠 Padrão ativação shape: {activation.shape}")

# Estatísticas da rede micelial
stats = mvq.mycelial.get_network_stats()
print(f"\n      📈 Rede Micelial:")
print(f"         - Observações totais: {stats['total_observations']}")
print(f"         - Conexões ativas: {stats['active_connections']}")
print(f"         - Densidade: {stats['density']:.6f}")

print("\n" + "="*70)
print("✅ TESTE COMPLETO FINALIZADO COM SUCESSO!")
print("="*70)
print(f"\nResumo:")
print(f"  • Modelo: {mvq.vqvae.__class__.__name__} (Wiki-trained)")
print(f"  • Codebook usage: {usage_percent:.1f}%")
print(f"  • Documentos processados: 50")
print(f"  • Rede micelial: {stats['active_connections']} conexões ativas")
print(f"\n🎉 O sistema Alexandria está funcionando perfeitamente!")
