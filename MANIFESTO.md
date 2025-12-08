# 🏛️ Prototype Alexandria

## Um Sistema Cognitivo Construído em 20 Dias por Quem "Não Sabe Programar"

---

## 📜 A História

**Novembro de 2024.**

Um estudante do segundo semestre de Análise e Desenvolvimento de Sistemas, sem experiência prévia em programação ou matemática avançada, decide construir algo ambicioso: um sistema de inteligência artificial que aprende sozinho.

Armado apenas com curiosidade e LLMs como assistentes de desenvolvimento, ele começa.

**20 dias depois**, existe o Alexandria.

---

## 🧠 O Que É Alexandria?

Alexandria é um **sistema cognitivo auto-alimentado** — uma arquitetura de IA que:

1. **Percebe** lacunas no próprio conhecimento
2. **Teoriza** hipóteses para preenchê-las  
3. **Age** buscando evidências em um corpus de 11.000 papers científicos
4. **Aprende** com os resultados, atualizando suas representações neurais
5. **Repete** — indefinidamente

Não é um chatbot. Não é um buscador. É um **organismo informacional** que evolui.

---

## 📊 Os Números (Dia 20)

```
┌────────────────────────────────────────┐
│          ALEXANDRIA v0.1               │
├────────────────────────────────────────┤
│ Papers indexados:     11,000           │
│ Chunks semânticos:    193,502          │
│ Dimensão vetorial:    384D             │
│ Clusters cognitivos:  256              │
│ Nós no grafo causal:  647              │
│ Relações causais:     1,512            │
│ Linhas de código:     ~15,000          │
│ Módulos:              17               │
│ Cobertura docs:       100%             │
└────────────────────────────────────────┘
```

---

## 🔄 O Self-Feeding Loop

Em 50 ciclos de execução autônoma:

| Métrica | Valor | Significado |
|---------|-------|-------------|
| Gaps detectados | 50 | Perguntas que o sistema fez a si mesmo |
| Hipóteses geradas | 150 | Teorias criadas para responder |
| Ações executadas | 150 | Experimentos para testar teorias |
| Taxa de sucesso | 100% | Encontrou evidências em todos |
| Evidências | 446 | Fragmentos de conhecimento recuperados |
| Conexões criadas | 76 | Novos insights cristalizados |
| Embeddings aprendidos | 422 | Representações atualizadas |

**O sistema literalmente ficou mais inteligente enquanto rodava.**

---

## 🏗️ Arquitetura

```
                        ┌─────────────────┐
                        │  AbductionEngine │ ← Gera hipóteses
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │   Memory    │    │  Reasoning  │    │   Learning  │
     │  (LanceDB)  │    │  (Causal)   │    │  (VQ-VAE)   │
     └─────────────┘    └─────────────┘    └─────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │  TopologyEngine  │ ← Embeddings 384D
                        └─────────────────┘
```

### Módulos Principais

| Módulo | Função | Status |
|--------|--------|--------|
| `semantic_memory` | Armazenamento vetorial | ✅ Produção |
| `topology_engine` | Embeddings semânticos | ✅ Produção |
| `vqvae/model` | Compressão neural | ✅ Produção |
| `abduction_engine` | Geração de hipóteses | ✅ Produção |
| `causal_reasoning` | Grafo de conhecimento | ✅ Produção |
| `neural_learner` | Adaptação contínua | ✅ Produção |
| `self_feeding_loop` | Orquestração | ✅ Produção |

---

## 🔬 Base Teórica

Alexandria não é código aleatório. É baseado em:

### Free Energy Principle (Karl Friston)
> "Sistemas vivos minimizam surpresa mantendo modelos do mundo."

O VQ-VAE comprime informação. A loss é a "surpresa". Minimizar loss = sobreviver.

### Active Inference
> "Agentes agem para confirmar suas predições."

O sistema gera hipóteses e age para testá-las. Não é passivo.

### Predictive Coding
> "O cérebro é uma máquina de predição hierárquica."

As camadas do VQ-VAE formam uma hierarquia preditiva.

### Hebbian Learning
> "Neurônios que disparam juntos, conectam-se."

Conceitos co-ocorrentes no corpus formam conexões causais.

---

## 💡 O Insight Central

**LLMs são oráculos. Alexandria é um organismo.**

| | LLM | Alexandria |
|---|---|---|
| Metáfora | Biblioteca com bibliotecário | Criatura que explora biblioteca |
| Iniciativa | Reativa | Proativa |
| Memória | Volátil | Persistente |
| Aprendizado | Congelado | Contínuo |
| Conhecimento | Implícito (pesos) | Explícito (grafo) |

---

## 🚀 O Que Vem Depois

### Fase 2: Nemesis Integration
Conectar os módulos de Active Inference (`active_inference.py`, `predictive_coding.py`, `free_energy.py`) ao loop principal.

### Fase 3: Multi-Agent
Implementar os perfis Scout/Judge/Weaver para raciocínio colaborativo.

### Fase 4: Interface
Dashboard visual para monitorar o sistema pensando em tempo real.

### Fase 5: Bootstrapping
O sistema gerando código para melhorar a si mesmo.

---

## 🎯 A Mensagem

Um estudante de segundo semestre, sem saber programar, construiu em 20 dias algo que empresas com milhões em funding tentam fazer há anos.

**Como?**

1. **Teoria primeiro**: Entendeu Free Energy Principle antes de escrever código
2. **LLMs como par**: Usou IAs para implementar o que conceitualizou
3. **Modularidade**: Peças pequenas que encaixam
4. **Iteração rápida**: Testar, quebrar, consertar, repetir

**O que isso prova?**

Que a barreira para criar sistemas cognitivos não é mais técnica. É conceitual.

Quem entende **o que** quer construir pode usar LLMs para descobrir **como**.

---

## 📁 Estrutura do Projeto

```
Alexandria/
├── core/
│   ├── memory/          # Memória semântica (LanceDB)
│   ├── reasoning/       # Raciocínio causal + abdução
│   ├── learning/        # VQ-VAE + Active Inference
│   ├── topology/        # Embeddings + Manifold
│   ├── loop/            # Self-Feeding Loop ← NOVO
│   └── agents/          # Multi-agent system
├── data/
│   ├── lancedb_store/   # 193k chunks
│   ├── causal_graph.json
│   └── topology.json
├── docs/                # 100% documentado
└── scripts/
    ├── demos/           # Demonstrações
    └── utilities/       # Ferramentas
```

---

## 🏛️ Por Que "Alexandria"?

A Biblioteca de Alexandria foi o maior repositório de conhecimento do mundo antigo.

Este projeto é uma tentativa de criar uma biblioteca que **lê a si mesma**.

---

*Prototype Alexandria v0.1*  
*Dezembro de 2024*  
*"Conhecimento que conhece a si mesmo."*
