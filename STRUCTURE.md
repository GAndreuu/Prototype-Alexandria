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
│   │   ├── bridge_agent.py      # Bridge Metacognitivo
│   │   └── critic_agent.py      # Crítico de Hipóteses
│   │
│   ├── field/                    # [Beta] Cognição Geométrica
│   │   ├── manifold.py          # Espaço vetorial dinâmico
│   │   ├── metric.py            # Métrica Riemanniana
│   │   └── pre_structural_field.py # Wrapper principal
│   │
│   ├── integrations/             # [NEW] Integrações Unificadas
│   │   ├── alexandria_unified.py # Master Integration (AlexandriaCore)
│   │   ├── geodesic_bridge_integration.py
│   │   ├── nemesis_bridge_integration.py
│   │   ├── learning_field_integration.py
│   │   ├── abduction_compositional_integration.py
│   │   ├── agents_compositional_integration.py
│   │   └── loop_compositional_integration.py
│   │
│   ├── learning/                 # [Prod] Nemesis Core
│   │   ├── active_inference.py  # Agentes FEP
│   │   └── predictive_coding.py # Hierarquia Preditiva
│   │
│   ├── loop/                     # [Prod] Autonomia
│   │   ├── self_feeding_loop.py # Orquestrador
│   │   ├── action_selection.py  # Protocolo de Ação
│   │   ├── active_inference_adapter.py # Adaptador AI
│   │   └── hypothesis_executor.py # Executor Grounded
│   │
│   ├── memory/                   # [Prod] Memória Semântica
│   │   ├── storage.py           # LanceDB Wrapper
│   │   └── semantic_memory.py   # Sistema de Indexação
│   │
│   ├── reasoning/                # [Prod] Motores de Raciocínio
│   │   ├── mycelial_reasoning.py # Rede Hebbiana
│   │   ├── symbol_grounding.py   # Text -> Node Grounding
│   │   └── vqvae/               # Compressão Neural (Monolith)
│   │
│   ├── topology/                 # [Prod] Gestão de Espaço
│   │   └── topology_engine.py   # Clustering e Mapeamento
│   │
│   └── utils/                    # Utilitários Compartilhados
│
├── 📂 scripts/                   # FERRAMENTAS & OPERAÇÃO
│   ├── system_runner_v2.py       # → Executor do Sistema (Principal)
│   ├── entrypoint.sh            # → Docker entrypoint
│   ├── ingestion/               # [NEW] Ingestão de Dados
│   │   ├── ingest_incremental.py
│   │   ├── mass_arxiv_ingest.py
│   │   └── multi_api_ingest.py
│   ├── maintenance/              # → Saúde e Limpeza
│   ├── analysis/                 # → Ciência de Dados (~25 scripts)
│   ├── testing/                  # → Validação e Stress Tests
│   ├── debug/                   # → Diagnóstico (~11 scripts)
│   ├── demos/                   # → Demonstrações
│   ├── training/                # → Treino de Modelos
│   ├── utilities/               # → Helpers
│   ├── benchmarks/              # → Performance
│   ├── calibration/             # → Calibração
│   └── diagnostics/             # → Diagnósticos Profundos
│
├── 📂 tests/                     # [REORGANIZED] TESTES AUTOMATIZADOS
│   ├── conftest.py              # Fixtures pytest
│   ├── test_*.py                # 19 arquivos de teste
│   └── data/                    # Dados de teste
│
├── 📂 interface/                 # [NEW] UI STREAMLIT
│   ├── app.py                   # Entrada principal
│   └── pages/                   # Páginas multipage
│
├── 📂 docs/                      # BASE DE CONHECIMENTO
│   ├── concepts/                 # [High Value] Teoria Profunda
│   ├── core/                     # Manuais Técnicos (~40 arquivos)
│   │   ├── integrations/        # [NEW] Docs de Integração
│   │   └── ...
│   └── reports/                  # Relatórios Gerados
│
├── 📂 .agent/                    # PROTOCOLO OPERACIONAL
│   └── workflows/                # /slash-commands
│
├── 📂 .prompts/                  # INSTRUÇÕES DE LLM
│
└── 📂 data/                      # PERSISTÊNCIA DE ESTADO
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
