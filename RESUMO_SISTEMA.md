# 📚 ALEXANDRIA: SYSTEM CONTEXT ANTHOLOGY (V2 - COMPREHENSIVE)

> **META-INSTRUÇÃO PARA AGENTES IA**: Este documento é uma fusão completa de TODA a documentação do sistema Alexandria. Ele substitui a necessidade de acessar a pasta `docs/`. Cada seção abaixo corresponde a um arquivo de documentação real.

---

# 🏛️ CAPÍTULO 1: VISÃO & ESTRUTURA (ROOT)

## 📄 `README.md` (A Visão)
**Resumo**: Alexandria é uma **Arquitetura Cognitiva Sinergética** e **Biocameral**. Ela separa memória bruta (LanceDB) de raciocínio (Mycelial Network). Seus 3 pilares são:
1.  **Raciocínio Micelial**: Aprendizado Hebbiano (persistência de conexões).
2.  **Cognição Geométrica**: Pensamento como deformação de espaço (Riemannian Manifold).
3.  **Autonomia Ativa**: Self-Feeding Loop (sonho e auto-correção).
**Status**: Operacional.

## 📄 `STRUCTURE.md` (O Território)
**Resumo**: Mapeamento da árvore de diretórios.
- `core/`: O código fonte principal (Agents, Field, Learning, Loop, Memory, Reasoning).
- `scripts/`: Ferramentas de operação (Ingestão, Runner, Manutenção).
- `.agent/`: Protocolos operacionais e workflows.
- `docs/`: A base de conhecimento original.

---

# 💡 CAPÍTULO 2: CONCEITOS TEÓRICOS (`docs/concepts/`)

## 📄 `geometric_cognition.md`
**Resumo**: Define o "Campo Pré-Estrutural". O sistema não usa apenas vetores estáticos, mas um **Dynamic Manifold** que se expande e contrai. O pensamento é a navegação por **geodésicas** (menor resistência) neste espaço curvo, onde tópicos densos têm "gravidade" alta.

## 📄 `active_autonomy.md`
**Resumo**: Define o "Self-Feeding Loop". O sistema transforma o ciclo passivo (Input→Output) em ativo (Input→Sonho→Ação). Usa agentes (Scout, Weaver, Critic) para detectar gaps de conhecimento e gerar hipóteses automaticamente.

## 📄 `cognitive_resilience.md`
**Resumo**: Explica a persistência da "Mente" mesmo após "Lobotomia" (Memory Wipe). Devido à quantização determinística (VQ-VAE), a rede micelial retém as conexões entre conceitos mesmo se os textos originais forem deletados. "Esquece onde leu, lembra o que aprendeu".

---

# 🧠 CAPÍTULO 3: CORE DOCUMENTATION (`docs/core/`)

## 🤖 SEÇÃO: AGENTS (`core/agents/`)

### 📄 `action_agent.md`
**Resumo**: O "braço" do sistema. Executa ações (`ActionType`) como: Ajuste de parämetros, Rodar simulações, Chamadas de API. Possui um `SecurityController` (rate limit) e `ParameterController` (segurança de estado).

### 3. Otimização de Hardware (i9 + RX 580)
- **Manifold**: 32 dimensões (reduzido de 384 via PCA) para cálculo geodésico em CPU.
- **LLM**: Desativado localmente para economia de recursos.
- **Geodesic Flow**: Otimizado para execução em CPU com projeção dimensional.

### 4. Interface
- **Streamlit**: Dashboard interativo para visualização de estados.

### 📄 `bridge_agent.md`
**Resumo**: O agente metacognitivo. Identifica `KnowledgeGap` (o que não sei) e cria `BridgeRequest` (planos de pesquisa) para preenchê-los. Avalia se novos dados realmente conectam conceitos isolados.

### 📄 `critic_agent.md`
**Resumo**: A "consciência". Usa Gemini para avaliar hipóteses. Gera `TruthScore` (veracidade) e `RiskLevel` (segurança). Implementa auto-regulação: se aprovar demais, diminui a temperatura do sistema.

### 📄 `neural_oracle.md`
**Resumo**: O sintetizador híbrido. Usa "Cortex of Experts": Tático (TinyLlama local, rápido/privado) e Estratégico (Gemini cloud, complexo). Realiza "Semantic Collision" (fusão de ideias).

---

## 🌌 SEÇÃO: FIELD (`core/field/`)

### 📄 `README.md` (Field Overview)
**Resumo**: Wrapper que unifica geometria diferencial e VQ-VAE. Metáfora: "Gravidade Cognitiva". Componentes: Manifold, Métrica, Energia Livre.

### 📄 `manifold.md`
**Resumo**: `DynamicManifold`. Um espaço vetorial que pode adicionar/remover dimensões dinamicamente. Mantém pontos âncora (códigos VQ-VAE) para estruturar o espaço.

### 📄 `metric.md`
**Resumo**: `RiemannianMetric`. Calcula distâncias não-euclideanas. Implementa deformação local: $g_{ij}(x) = \delta_{ij} + \sum w \cdot \exp(-r^2)$.

### 📄 `free_energy_field.md`
**Resumo**: `FreeEnergyField`. Calcula $F(x) = E(x) - TS(x)$. Encontra atratores (mínimos locais) que representam conceitos estáveis.

### 📄 `geodesic_flow.md`
**Resumo**: `GeodesicFlow`. Resolve a equação geodésica $\ddot{x} + \Gamma \dot{x}\dot{x} = 0$. Simula o fluxo de pensamento seguindo a curvatura do campo.

### 📄 `cycle_dynamics.md`
**Resumo**: `CycleDynamics`. O ciclo cardíaco do campo: Expansão (novas dims) → Configuração (annealing) → Compressão (cristalização em grafo).

---

## 🎓 SEÇÃO: LEARNING (`core/learning/`)

### 📄 `active_inference.md`
**Resumo**: Agente baseado em Friston. Minimiza `Expected Free Energy` ($G = Risk + Ambiguity$). Escolhe ações epistêmicas (explorar) para reduzir incerteza e pragmáticas (explotar) para atingir objetivos.

### 📄 `predictive_coding.md`
**Resumo**: Rede hierárquica (Input → L1 → L2 → Code). Propaga **Erro** para cima e **Predição** para baixo. Aprendizado ocorre minimizando o erro de predição localmente.

### 📄 `meta_hebbian.md`
**Resumo**: Plasticidade evolutiva. Não aprende apenas pesos, mas a **regra de atualização** ($\Delta w = \eta(A \cdot pre \cdot post + ...)$). Usa estratégias evolutivas para otimizar a regra ABCD.

### 📄 `free_energy.md`
**Resumo**: Métrica unificadora. `VariationalFreeEnergy` para percepção ($F = Complexity - Accuracy$) e `ExpectedFreeEnergy` para ação.

### 📄 `integration_layer.md`
**Resumo**: Glue code. Resolve conflitos entre módulos (ex: adapta matrizes densas do Meta-Hebbian para grafos esparsos do Mycelial). Gerencia Resource Profiles (LITE, BALANCED, PERFORMANCE).

### 📄 `NEMESIS_MANUAL.md`
**Resumo**: Manual do subsistema "Cognitive Nemesis". Define 3 personas: Scout (Explorador), Judge (Crítico), Weaver (Conector). Otimizado para hardware de consumo (Lite Mode).

---

## 🔄 SEÇÃO: LOOP (`core/loop/`)

### 📄 `self_feeding_loop.md`
**Resumo**: Orquestrador principal. Loop contínuo: Detectar Gaps → Gerar Hipóteses → Executar Ações → Coletar Feedback → Atualizar Modelos.

### 📄 `nemesis_integration.md`
**Resumo**: Cérebro executivo do loop. Seleciona a melhor ação baseada em EFE. Fecha o ciclo de feedback atualizando o modelo generativo com recompensas.

### 📄 `active_inference_adapter.md`
**Resumo**: Adaptador que conecta a teoria da Active Inference (FEP) com o loop pragmático. Implementa o protocolo `ActionSelectionAdapter` para permitir que o sistema alterne entre heurísticas e inferência profunda.

### 📄 `action_selection.md`
**Resumo**: Protocolo unificado de tipos de ação (`QUERY_SEARCH`, `BRIDGE_CONCEPTS`, etc.). Define o contrato para qualquer agente que queira controlar o corpo do Alexandria.

---

## 💾 SEÇÃO: MEMORY (`core/memory/`)

### 📄 `semantic_memory.md`
**Resumo**: `SemanticFileSystem`. Gerencia indexação multimodal. Pipeline: PDF/Imagem → Router → Chunking Inteligente → Embedding 384D → LanceDB.

### 📄 `storage.md`
**Resumo**: Wrapper do LanceDB. Garante persistência eficiente, busca vetorial e armazenamento de metadados.

### 📄 `v11_vision_encoder.md`
**Resumo**: Encoder visual hierárquico. Transforma imagens em vetores 384D compatíveis com o espaço semântico de texto.

---

## 🔬 SEÇÃO: REASONING (`core/reasoning/`)

### 📄 `vqvae.md`
**Resumo**: "O Codec do Cérebro". Comprime vetores 384D em 4 códigos discretos (4 bytes). Permite que o sistema manipule conceitos abstratos simbolicamente. Modelo atual: MonolithWiki (96% compressão).

### 📄 `mycelial_reasoning.md`
**Resumo**: "A Rede Neural". Grafo esparso onde nós são pares (Head, Code). Aprendizado Hebbiano ("Fire together, wire together"). Raciocínio é a propagação de ativação neste grafo.

### 📄 `abduction_engine.md`
**Resumo**: Motor de hipóteses. Detecta 3 tipos de gaps: Cluster Órfão, Conexão Ausente, Corrente Quebrada. Gera hipóteses usando templates e valida via coerência semântica.

### 📄 `causal_reasoning.md`
**Resumo**: Grafo Causal. Tenta inferir direção (A causa B) usando padrões temporais em textos e verbos causais. Detecta variáveis latentes (causas ocultas de correlação).

### 📄 `symbol_grounding.md`
**Resumo**: O elo perdido entre texto e grafo. Converte strings arbitrárias ("autonomy") em códigos VQ-VAE concretos ((Head, Code)). Permite que o executor realize ações precisas no grafo baseadas em comandos abstratos.

---

## 🗺️ SEÇÃO: TOPOLOGY (`core/topology/`)

### 📄 `topology_engine.md`
**Resumo**: Gerenciador do espaço 384D. Wrapper do `sentence-transformers`. Realiza Clustering (K-Means) e Redução de Dimensionalidade (UMAP/PCA).

---

## 🛠️ SEÇÃO: UTILS (`core/utils/`)

### 📄 `README.md` (Utils)
**Resumo**:
- **Harvester**: Scraper de Arxiv.
- **LocalLLM**: TinyLlama-1.1B para inferência rápida na CPU.
- **Logger**: Loguru estruturado.

---

## 🔌 SEÇÃO: INTEGRATIONS (`core/integrations/`) - [NEW]

### 📄 `alexandria_unified.md`
**Resumo**: `AlexandriaCore` - Fachada unificada. Um único ponto de entrada para executar ciclos cognitivos completos (Perceive→Reason→Act→Learn). Coordena Geodesic, Nemesis, Abduction, Agents e Loop.

### 📄 `geodesic_bridge_integration.md`
**Resumo**: Integra o motor de fluxo geodésico ao manifold curvo. Permite computar caminhos semânticos (geodésicas) entre conceitos respeitando a curvatura do espaço.

### 📄 `nemesis_bridge_integration.md`
**Resumo**: Conecta Active Inference ao manifold. O EFE (Expected Free Energy) agora é calculado via distância geodésica, tornando o agente ciente da topologia.

### 📄 `learning_field_integration.md`
**Resumo**: Unifica PC, AI e Meta-Hebbian com o campo. Erros de predição são geodésicos, planejamento usa EFE curvo, learning rates dependem da curvatura local.

### 📄 `abduction_compositional_integration.md`
**Resumo**: Representa gaps como descontinuidades geométricas e hipóteses como caminhos geodésicos que fecham esses gaps.

### 📄 `agents_compositional_integration.md`
**Resumo**: Enriquece todos os agentes (Action, Bridge, Critic, Oracle) com consciência geométrica.

### 📄 `loop_compositional_integration.md`
**Resumo**: Fecha o ciclo autônomo. Feedback agora deforma o manifold, tornando caminhos de sucesso mais fáceis de percorrer.

---

# ⚙️ CAPÍTULO 4: PROTOCOLOS (.agent/workflows/)

## 📄 `onboarding.md`
**Resumo**: Workflow de "Total Recall". O agente lê todos os arquivos `.md` do projeto para carregar contexto total antes de começar a trabalhar.

## 📄 `criar-feature.md`
**Resumo**: Workflow de Scaffold. Cria automaticamente a estrutura de arquivos (`core/`, `tests/`, `docs/`) para uma nova feature, garantindo padronização.

## 📄 `documentar-projeto.md`
**Resumo**: Workflow de documentação. Analisa código não documentado e gera arquivos `.md` correspondentes.

## 📄 `review-completo.md`
**Resumo**: Workflow de CI/CD manual. Roda testes, linter, verifica segurança e gera relatório antes de merges.

---

# 🧪 CAPÍTULO 5: VALIDAÇÃO

## 📄 `scripts/validate_alexandria.md`
**Resumo**: Documentação do script de prova de conceito. Compara o algoritmo Alexandria (Field+Mycelial) contra um Baseline (K-Means). Métricas: Pureza de Cluster, Recuperação de Conexões, Desvio Geodésico.
