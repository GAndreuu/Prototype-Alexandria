#!/usr/bin/env python3
"""
Gerador de Relatório Visual - Self-Feeding Loop
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurar estilo
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Carregar métricas
metrics_path = Path("data/real_loop_metrics.json")
with open(metrics_path) as f:
    data = json.load(f)

cycles = data.get('cycles', [])
print(f"Ciclos carregados: {len(cycles)}")

if not cycles:
    print("Nenhum ciclo encontrado!")
    exit(1)

# Extrair dados por ciclo
cycle_nums = [c.get('cycle_id', i) for i, c in enumerate(cycles)]
gaps = [c.get('gaps_detected', 0) for c in cycles]
hypotheses = [c.get('hypotheses_generated', 0) for c in cycles]
actions = [c.get('actions_executed', 0) for c in cycles]
successes = [c.get('actions_successful', 0) for c in cycles]
evidences = [c.get('total_evidence', 0) for c in cycles]
connections = [c.get('new_connections', 0) for c in cycles]
rewards = [c.get('avg_reward', 0) for c in cycles]

# Calcular acumulados
cum_evidences = np.cumsum(evidences)
cum_connections = np.cumsum(connections)
cum_successes = np.cumsum(successes)

# Success rate rolling
window = 10
success_rate = []
for i in range(len(cycles)):
    start = max(0, i - window + 1)
    total_actions = sum(actions[start:i+1])
    total_success = sum(successes[start:i+1])
    rate = total_success / max(1, total_actions)
    success_rate.append(rate * 100)

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('📊 Alexandria Self-Feeding Loop - Relatório de 100 Ciclos', fontsize=16, fontweight='bold')

# 1. Evidências e Conexões Acumuladas
ax1 = axes[0, 0]
ax1.fill_between(cycle_nums, cum_evidences, alpha=0.3, color='cyan', label='Evidências')
ax1.plot(cycle_nums, cum_evidences, 'c-', linewidth=2)
ax1.fill_between(cycle_nums, cum_connections, alpha=0.3, color='magenta', label='Conexões')
ax1.plot(cycle_nums, cum_connections, 'm-', linewidth=2)
ax1.set_xlabel('Ciclo')
ax1.set_ylabel('Quantidade Acumulada')
ax1.set_title('🔗 Evidências e Conexões Acumuladas')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# 2. Taxa de Sucesso (Rolling)
ax2 = axes[0, 1]
ax2.plot(cycle_nums, success_rate, 'g-', linewidth=2, label=f'Taxa de Sucesso (janela={window})')
ax2.axhline(y=100, color='lime', linestyle='--', alpha=0.5, label='100%')
ax2.fill_between(cycle_nums, success_rate, alpha=0.2, color='green')
ax2.set_xlabel('Ciclo')
ax2.set_ylabel('Taxa de Sucesso (%)')
ax2.set_title('✅ Taxa de Sucesso ao Longo do Tempo')
ax2.set_ylim(0, 110)
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

# 3. Reward por Ciclo
ax3 = axes[1, 0]
colors = ['green' if r > 0 else 'red' for r in rewards]
ax3.bar(cycle_nums, rewards, color=colors, alpha=0.7, width=0.8)
ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
ax3.set_xlabel('Ciclo')
ax3.set_ylabel('Reward')
ax3.set_title('🎯 Reward por Ciclo')
ax3.grid(alpha=0.3)

# 4. Hipóteses vs Ações
ax4 = axes[1, 1]
ax4.bar(cycle_nums, hypotheses, alpha=0.5, color='orange', label='Hipóteses', width=0.8)
ax4.bar(cycle_nums, actions, alpha=0.5, color='blue', label='Ações', width=0.4)
ax4.set_xlabel('Ciclo')
ax4.set_ylabel('Quantidade')
ax4.set_title('💡 Hipóteses Geradas vs Ações Executadas')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()

# Salvar figura
output_path = Path("/home/G/.gemini/antigravity/brain/d24652a0-7d31-4565-9e04-fa790d316795/loop_metrics_chart.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Gráfico salvo em: {output_path}")

# Estatísticas finais
print("\n" + "="*60)
print("📊 ESTATÍSTICAS FINAIS")
print("="*60)
print(f"Total de Ciclos: {len(cycles)}")
print(f"Total de Evidências: {sum(evidences)}")
print(f"Total de Conexões: {sum(connections)}")
print(f"Taxa de Sucesso Geral: {sum(successes)/max(1,sum(actions))*100:.1f}%")
print(f"Reward Médio: {np.mean(rewards):.4f}")
print(f"Reward Total: {sum(rewards):.4f}")
