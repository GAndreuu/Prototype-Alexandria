# 🗺️ Alexandria - Estrutura do Projeto

**Visão Topográfica Atualizada**
> Este documento mapeia a anatomia completa do sistema Alexandria.

---

## 🏗️ Árvore de Diretórios

```
Alexandria/
├── 📂 core/                      # O NÚCLEO COGNITIVO
│   ├── agents/                   # [Alpha] Sistema de Agentes
│   │   ├── action/              # Agentes de Ação (V2)
│   │   └── critic_agent.py      # Crítico de Hipóteses
│   │
│   ├── field/                    # [Beta] Cognição Geométrica
│   │   ├── manifold.py          # Espaço vetorial dinâmico
│   │   ├── metric.py            # Métrica Riemanniana
│   │   └── pre_structural_field.py # Wrapper principal
│   │
│   ├── learning/                 # [Prod] Nemesis Core
│   │   ├── active_inference.py  # Agentes FEP
│   │   └── predictive_coding.py # Hierarquia Preditiva
│   │
│   ├── loop/                     # [Beta] Autonomia
│   │   ├── self_feeding_loop.py # Orquestrador de Sonhos
│   │   └── nemesis_integration.py # Ponte Cérebro-Mente
│   │
│   ├── memory/                   # [Prod] Memória Semântica
│   │   ├── storage.py           # LanceDB Wrapper
│   │   └── semantic_memory.py   # Sistema de Indexação
│   │
│   ├── reasoning/                # [Prod] Motores de Raciocínio
│   │   ├── mycelial_reasoning.py # Rede Hebbiana
│   │   └── vqvae/               # Compressão Neural (Monolith)
│   │
│   ├── topology/                 # [Prod] Gestão de Espaço
│   │   └── topology_engine.py   # Clustering e Mapeamento
│   │
│   └── utils/                    # Utilitários Compartilhados
│
├── 📂 scripts/                   # FERRAMENTAS & OPERAÇÃO
│   ├── mass_arxiv_ingest.py      # → Ingestão de Papers (Principal)
│   ├── system_runner_v2.py       # → Executor do Sistema (Principal)
│   ├── maintenance/              # → Saúde e Limpeza
│   │   ├── check_mycelial_health.py
│   │   └── prune_mycelial.py
│   ├── analysis/                 # → Ciência de Dados
│   │   ├── alexandria_topics.py
│   │   └── geometric_topics.py
│   └── testing/                  # → Validação
│       └── validate_alexandria.py
│
├── 📂 docs/                      # BASE DE CONHECIMENTO
│   ├── concepts/                 # [High Value] Teoria Profunda
│   │   ├── active_autonomy.md
│   │   ├── geometric_cognition.md
│   │   └── cognitive_resilience.md
│   ├── core/                     # Manuais Técnicos
│   └── reports/                  # Relatórios Gerados
│
├── 📂 .agent/                    # PROTOCOLO OPERACIONAL
│   └── workflows/                # /slash-commands
│       ├── onboarding.md        # /onboarding (Total Recall)
│       ├── criar-feature.md     # /criar-feature
│       ├── documentar-projeto.md # /documentar-projeto
│       ├── review-completo.md   # /review-completo
│       └── debug-profundo.md    # /debug-profundo
│
├── 📂 .prompts/                  # INSTRUÇÕES DE LLM
│   ├── analisar_estrutura.md    # Prompt de Análise + Reality Check
│   └── ...
│
├── 📂 modulo_operacional/        # CONCEITOS & DESIGN
│   └── Cosmic Garden...md       # Inspiração para arquitetura de agentes
│
└── � data/                      # PERSISTÊNCIA DE ESTADO
    ├── library/                 # PDFs crus
    ├── lancedb_store/           # Vetores (Memória Episódica)
    ├── mycelial_state.npz       # Grafo (Raciocínio Persistente)
    └── monolith_v13_trained.pth # Modelo VQ-VAE
```

---

## 🔍 Detalhes dos Módulos Principais

### 1. `core/field` (Cognição Geométrica)
Implementa a ideia de que pensar é deformar o espaço.
- **Status**: Beta
- **Arquivos Chave**: `metric.py` (calcula distâncias curvas), `geodesic_flow.py` (encontra conexões não-lineares).

### 2. `core/loop` (Autonomia)
O mecanismo que permite ao sistema operar sem usuário.
- **Status**: Beta
- **Fluxo**: Observar Grafo → Detectar Gaps → Gerar Hipótese → Validar → Consolidar.

### 3. `core/reasoning/vqvae` (Compressão Neural)
O coração da eficiência do Alexandria.
- **Status**: Produção
- **Specs**: Reduz vetores 384D para apenas 4 bytes com perda mínima. Permite rodar grafos gigantes em hardware modesto.

---

## � Estatísticas de Código (Estimada)

- **Python**: ~20k linhas
- **Módulos Core**: 8
- **Scripts Utilitários**: 15+
- **Documentação**: ~30 arquivos Markdown

---

## 🛠️ Onde encontrar o que você precisa?

| Eu quero... | Vá para... |
|-------------|------------|
| Iniciar o sistema | `scripts/system_runner_v2.py` |
| Ingerir dados | `scripts/mass_arxiv_ingest.py` |
| Entender a teoria | `docs/concepts/` |
| Criar nova feature | `.agent/workflows/criar-feature.md` |
| Checar saúde | `scripts/maintenance/` |
| Debugar | `.agent/workflows/debug-profundo.md` |
