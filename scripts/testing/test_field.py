#!/usr/bin/env python3
"""
Teste do PreStructuralField (Wrapper Unificado)
================================================

Testa a integração completa do Campo Pré-Estrutural.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

print("=" * 60)
print("🧪 TESTE: PreStructuralField (Wrapper Unificado)")
print("=" * 60)

# ============================================================
# 1. Teste Básico
# ============================================================
print("\n📦 [1/4] Importando PreStructuralField...")

try:
    from core.field import PreStructuralField, PreStructuralConfig
    print("   ✅ Import OK")
except Exception as e:
    print(f"   ❌ Import FALHOU: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 2. Inicialização
# ============================================================
print("\n🚀 [2/4] Inicializando com dimensão reduzida...")

try:
    config = PreStructuralConfig(
        base_dim=32,  # Reduzido para teste rápido
        max_expansion=16,
        configuration_steps=5,
        max_geodesic_steps=5
    )
    
    field = PreStructuralField(config)
    
    print(f"   ✅ PreStructuralField inicializado")
    print(f"      - Dimensão: {config.base_dim}")
    print(f"      - Temperatura: {config.temperature}")
    
except Exception as e:
    print(f"   ❌ Inicialização FALHOU: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 3. Trigger & Propagate
# ============================================================
print("\n⚡ [3/4] Testando trigger e propagação...")

try:
    # Cria embeddings de teste
    for i in range(3):
        emb = np.random.randn(32)
        emb = emb / np.linalg.norm(emb)
        
        state = field.trigger(emb, intensity=0.5 + i*0.2)
        print(f"      Trigger {i+1}: F_mean = {state.mean_free_energy:.4f}, atratores = {state.num_attractors}")
    
    # Propaga
    states = field.propagate(steps=3)
    
    print(f"   ✅ Trigger & Propagate OK")
    print(f"      - Estados gerados: {len(states)}")
    
except Exception as e:
    print(f"   ❌ Trigger/Propagate FALHOU: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 4. Crystallize
# ============================================================
print("\n💎 [4/4] Testando cristalização...")

try:
    graph = field.crystallize()
    
    print(f"   ✅ Crystallize OK")
    print(f"      - Nós: {len(graph['nodes'])}")
    print(f"      - Arestas: {len(graph['edges'])}")
    
except Exception as e:
    print(f"   ❌ Crystallize FALHOU: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Stats
# ============================================================
print("\n" + "=" * 60)
print("📊 ESTATÍSTICAS")
print("=" * 60)

try:
    stats = field.stats()
    print(f"""
   Manifold:
     - Dimensão atual: {stats['manifold']['current_dim']}
     - Pontos: {stats['manifold']['num_points']}
     - Ativos: {stats['manifold']['active_points']}
   
   Metric:
     - Deformações: {stats['metric']['deformations']}
   
   Field:
     - Temperatura: {stats['field']['temperature']}
     - Atratores: {stats['field']['num_attractors']}
   
   Triggers: {stats['triggers']}
   Ciclos: {stats['cycles_completed']}
   
   Conexões:
     - VQ-VAE: {stats['connected']['vqvae']}
     - Mycelial: {stats['connected']['mycelial']}
     - FreeEnergy: {stats['connected']['variational_fe']}
""")
except Exception as e:
    print(f"   Stats FALHOU: {e}")

print("=" * 60)
print("✅ TESTE CONCLUÍDO")
print("=" * 60)
