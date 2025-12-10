# -*- coding: utf-8 -*-
"""
Script to inspect Hebbian connections created by crystallization.
"""
import sys
sys.path.insert(0, '.')

import logging
logging.disable(logging.CRITICAL)  # Suppress all logging

import numpy as np
from core.field.pre_structural_field import PreStructuralField
from core.reasoning.mycelial_reasoning import MycelialReasoning, MycelialConfig

# Init
mc = MycelialConfig(save_path='/tmp/inspect_conn.pkl')
m = MycelialReasoning(mc)
m.reset()
f = PreStructuralField()
f.connect_mycelial(m)

# Trigger 5 pontos com codigos distintos
codes_list = []
for i in range(5):
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    codes = np.array([10+i, 20+i, 30+i, 40+i], dtype=np.int32)
    codes_list.append(codes)
    f.trigger(embedding=emb, codes=codes, intensity=1.0)

print("=" * 60)
print("CODIGOS VQ-VAE DOS 5 PONTOS")
print("=" * 60)
for i, c in enumerate(codes_list):
    print(f"  Ponto {i}: {c.tolist()}")

# Crystallize
g = f.crystallize()

print()
print("=" * 60)
print("ARESTAS DO GRAFO CRISTALIZADO")
print("=" * 60)
for e in g['edges'][:10]:
    src = e["source"]
    tgt = e["target"]
    w = e["weight"]
    print(f"  {src} <-> {tgt} (peso: {w:.4f})")

print()
print("=" * 60)
print("CONEXOES HEBBIANAS NO MYCELIAL (Top 20)")
print("=" * 60)
conns = m.get_strongest_connections(20)
for c in conns:
    h_from, c_from = c['from']
    h_to, c_to = c['to']
    strength = c['strength']
    print(f"  Head {h_from}: code {c_from:3d} <-> Head {h_to}: code {c_to:3d} | strength: {strength:.4f}")

print()
print("=" * 60)
print("ANALISE: FAZ SENTIDO?")
print("=" * 60)
print("Os codigos dos pontos eram:")
for i, c in enumerate(codes_list):
    print(f"  P{i}: head0={c[0]}, head1={c[1]}, head2={c[2]}, head3={c[3]}")
print()
print("Conexoes deveriam ligar codigos de pontos proximos no manifold.")
print("Ex: se P0(10,20,30,40) e P1(11,21,31,41) estao proximos,")
print("    entao (head0, 10) <-> (head0, 11) deve existir.")

# Save to file
with open("data/connection_analysis.txt", "w") as f:
    f.write("ANALISE DE CONEXOES HEBBIANAS\n")
    f.write("=" * 60 + "\n\n")
    f.write("CODIGOS VQ-VAE DOS 5 PONTOS:\n")
    for i, c in enumerate(codes_list):
        f.write(f"  Ponto {i}: {c.tolist()}\n")
    f.write("\nARTESTAS DO GRAFO CRISTALIZADO:\n")
    for e in g['edges']:
        f.write(f"  {e['source']} <-> {e['target']} (peso: {e['weight']:.4f})\n")
    f.write("\nCONEXOES HEBBIANAS (Top 20):\n")
    for c in conns:
        h_from, c_from = c['from']
        h_to, c_to = c['to']
        f.write(f"  Head {h_from}: code {c_from:3d} <-> Head {h_to}: code {c_to:3d} | strength: {c['strength']:.4f}\n")
    
print("\nResultados salvos em data/connection_analysis.txt")
