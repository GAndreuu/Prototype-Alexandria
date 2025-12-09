# 🏛️ Alexandria - Synergetic Cognitive Architecture

<div align="center">

![Status](https://img.shields.io/badge/status-active_development-success?style=for-the-badge&color=2ea44f)
![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python)
![Architecture](https://img.shields.io/badge/architecture-biocameral-purple?style=for-the-badge)

**Arquitetura Cognitiva Sinergética para Raciocinio Local e Autônomo**

*Uma síntese de sistemas biológicos, geometria diferencial e inferência ativa.*

[Conceitos Chave](#-pilares-do-sistema) • [Realidade vs Aspiração](#-reality-check-o-que-funciona) • [Operação](#-protocolo-operacional) • [Quick Start](#-quick-start)

</div>

---

## 🎯 Visão Sintética

Alexandria não é apenas um RAG (Retrieval-Augmented Generation). É um **Sistema Cognitivo Biocameral** que separa memória (dados brutos) de raciocínio (grafo de conceitos), permitindo:

1.  **Resiliência Cognitiva**: O sistema "entende" conceitos mesmo se a memória bruta for apagada.
2.  **Geometria do Pensamento**: O espaço vetorial se deforma para aproximar conceitos logicamente conectados.
3.  **Sonho Autônomo**: Enquanto ocioso, o sistema cria novas conexões e hipóteses sozinho.

---

## 🏛️ Pilares do Sistema

### 1. 🍄 Raciocínio Micelial (The Mycelial Network)
Baseado no aprendizado Hebbiano (*"Cells that fire together, wire together"*). O sistema constrói um grafo de pesos sinápticos entre conceitos quantizados (tokens VQ-VAE), criando uma estrutura de longo prazo independente dos dados originais.
- [📄 Ler Conceito Completo](docs/concepts/cognitive_resilience.md)

### 2. 🌌 Cognição Geométrica (The Pre-Structural Field)
Baseado em Geometria Diferencial. O sistema mantém um *manifold* dinâmico onde a distância entre dois conceitos muda baseada na "gravidade" da informação acumulada. Pensar é navegar geodésicas (curvas de menor resistência) neste espaço.
- [📄 Ler Conceito Completo](docs/concepts/geometric_cognition.md)

### 3. 🧬 Autonomia Ativa (The Self-Feeding Loop)
Baseado em Active Inference e Abduction. Um loop contínuo que observa o próprio grafo, detecta "ilhas" de conhecimento isolado e tenta criar pontes lógicas (hipóteses) para conectá-las, sem intervenção humana.
- [📄 Ler Conceito Completo](docs/concepts/active_autonomy.md)

---

## 🧠 Reality Check: O que Funciona?

Para garantir transparência total, mantemos esta tabela de *Capabilities vs Aspirations*:

| Componente | Status | Realidade do Código | Evidência |
|:---|:---:|:---|:---|
| **VQ-VAE Monolith** | ✅ Prod | Compressão neural 96% funcional. Codebook 99% ativo. | `core/reasoning/vqvae/` |
| **Mycelial Network** | ✅ Prod | 600k+ conexões hebbianas. Persistência comprovada. | `core/reasoning/mycelial_reasoning.py` |
| **PreStructuralField** | ⚠️ Beta | Métrica Riemanniana implementada, otimização pendente. | `core/field/` |
| **SelfFeedingLoop** | ⚠️ Beta | Detecção de gaps funciona, geração de hipóteses básica. | `core/loop/` |
| **Active Inference** | ⚠️ Alpha | Agentes (Scout/Weaver) existem mas são rudimentares. | `core/learning/nemesis_agents.py` |
| **Meta-Consciousness** | ❌ Plan | Capacidade de auto-modificação de código ainda não existe. | N/A |

---

## ⚙️ Protocolo Operacional

Alexandria opera em um modo **Agentic First**. A interação principal não é apenas rodar scripts, mas orquestrar fluxos de trabalho.

### The Agentic Core (`.agent/`)
O sistema possui workflows autônomos acessíveis via comandos:

- **`/onboarding`**: O sistema lê toda a documentação ("Total Recall") e se situa.
- **`/criar-feature`**: Cria automaticamente a estrutura de pastas, classes e testes para novos módulos.
- **`/documentar-projeto`**: Varre o código, encontra falhas de documentação e escreve os manuais.
- **`/review-completo`**: Realiza auditoria de código, segurança e performance antes de merges.
- **`/debug-profundo`**: Rastreia dependências e fluxo de dados para resolver bugs complexos.

> **Nota**: Estes workflows residem em `.agent/workflows/` e são executados pelo agente principal.

---

## 🏗️ Estrutura do Código

```
Alexandria/
├── core/                  # O Cérebro
│   ├── field/             # → Cognição Geométrica (Riemmanian Manifold)
│   ├── loop/              # → Autonomia (Self-Feeding Loop)
│   ├── reasoning/         # → VQ-VAE e Mycelial Network
│   ├── memory/            # → LanceDB e SemanticFileSystem
│   └── agents/            # → Agentes especializados
│
├── scripts/               # Ferramentas
│   ├── mass_arxiv_ingest.py  # → Ingestão massiva de papers
│   ├── system_runner_v2.py   # → Loop principal do sistema
│   ├── maintenance/          # → Scripts de cura e limpeza
│   ├── analysis/             # → Ferramentas de diagnóstico
│   └── testing/              # → Scripts de validação
│
├── docs/                  # Conhecimento
│   ├── concepts/          # → Teoria profunda (Novos!)
│   ├── architecture/      # → Diagramas técnicos
│   └── reports/           # → Relatórios gerados pelo agente
│
└── .agent/                # Protocolos
    └── workflows/         # → Receitas de automação (/slash-commands)
```

---

## 🚀 Quick Start

### 1. Instalação
```bash
git clone https://github.com/GAndreuu/Prototype-Alexandria.git
cd Alexandria
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingestão de Dados (Alimentar o Cérebro)
```bash
# Baixa e processa papers do ArXiv sobre AI/ML
python scripts/mass_arxiv_ingest.py --max-papers 100
```

### 3. Iniciar o Sistema (Acordar a Mente)
```bash
# Inicia o loop autônomo e a API
python scripts/system_runner_v2.py
```

### 4. Modo Manutenção (Opcional)
```bash
# Checar saúde da rede micelial
python scripts/maintenance/check_mycelial_health.py
```

---

## 🤝 Contribuindo

Este é um projeto de pesquisa ativa.
- Use `/criar-feature` para adicionar funcionalidades.
- Use `/review-completo` antes de abrir PRs.
- Leia `docs/concepts/` antes de tocar no Core.

---

<div align="center">
    <b>Alexandria System</b><br>
    <i>Meta-Cognição Local</i>
</div>
