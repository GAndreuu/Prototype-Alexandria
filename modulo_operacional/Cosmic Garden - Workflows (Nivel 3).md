# 🔮 Cosmic Garden: Camada de Workflows (Nível 3)

**Extensão do protocolo base**  
**Versão**: 2.0  
**Tipo**: Sistema de Macros e Workflows Automatizados

---

## 🎯 Conceito: 3 Níveis de Abstração

```
┌─────────────────────────────────────────────────────────┐
│ NÍVEL 3: WORKFLOWS (NOVO!)                              │
│ Sequences complexas de ações                            │
│ Exemplo: (executar analisar_estrutura)                  │
│   → Carrega contextos                                   │
│   → Executa análise                                     │
│   → Gera relatório                                      │
│   → Salva resultados                                    │
├─────────────────────────────────────────────────────────┤
│ NÍVEL 2: CONTEXTOS COMPOSTOS (já existe)                │
│ Múltiplos arquivos agrupados                            │
│ Exemplo: (ativar contexto_docs)                         │
│   → Carrega 4 arquivos de documentação                  │
├─────────────────────────────────────────────────────────┤
│ NÍVEL 1: ALIASES SIMPLES (já existe)                    │
│ Referência a arquivo único                              │
│ Exemplo: (ativar vqvae)                                 │
│   → Carrega core/reasoning/vqvae/model.py               │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Estrutura de Workflows

### Arquivo: `.cosmic_garden/WORKFLOWS.md` (Conceito) 
> **Nota de Implementação**: No Alexandria, os workflows reais ficam em `.agent/workflows/*.md`.

```markdown
# 🔮 Workflows Automatizados

## Sintaxe

```
/workflow_name
```

Ou via chat:
```
(executar [workflow_name])
```

## Definição de Workflow

```yaml
workflow_name:
  description: "O que este workflow faz"
  contexts_required: [lista de contextos]
  steps:
    - step1: "ação"
    - step2: "ação"
    - step3: "ação"
  output_format: "formato esperado"
```
```

---

## 🔧 Workflows Padrão para Alexandria

### 1. Workflow: `analisar_estrutura`

```yaml
analisar_estrutura:
  description: "Análise completa da estrutura do projeto"
  
  contexts_required:
    - estrutura (STRUCTURE.md)
    - readme (README.md)
    - arch_tecnica (docs/architecture/technical.md)
  
  steps:
    1. Carregar contextos
       → Ler STRUCTURE.md, README.md, technical.md
    
    2. Analisar organização
       → Identificar pastas principais
       → Mapear módulos
       → Detectar dependências
    
    3. Avaliar consistência
       → Verificar se README reflete estrutura real
       → Verificar se STRUCTURE está atualizado
       → Detectar arquivos não documentados
    
    4. Gerar relatório
       → Listar módulos principais
       → Identificar áreas bem/mal documentadas
       → Sugerir melhorias
  
  output_format:
    type: "markdown_report"
    sections:
      - "Resumo Executivo"
      - "Estrutura Atual"
      - "Análise de Consistência"
      - "Recomendações"
  
  exemplo_uso:
    "(executar analisar_estrutura)"
```

**Saída esperada**:
```markdown
# Relatório de Análise Estrutural - Alexandria

## Resumo Executivo
Projeto bem organizado com 4 módulos principais...

## Estrutura Atual
- core/ (8 submódulos)
  - agents/ (4 arquivos)
  - learning/ (5 arquivos)
  ...

## Análise de Consistência
✅ README.md está atualizado
✅ STRUCTURE.md reflete pastas reais
⚠️ 3 arquivos novos não documentados:
  - core/learning/integration_layer.py
  ...

## Recomendações
1. Documentar integration_layer.py
2. Atualizar diagrama de arquitetura
...
```

---

### 2. Workflow: `atualizar_documentacao`

```yaml
atualizar_documentacao:
  description: "Atualiza documentação baseado em mudanças no código"
  
  contexts_required:
    - contexto_docs (toda documentação)
    - [módulo_modificado] (dinâmico)
  
  parameters:
    - target_module: "qual módulo foi modificado"
    - change_type: "novo|modificado|removido"
  
  steps:
    1. Detectar mudanças
       → Ler código atual do módulo
       → Comparar com documentação existente
       → Identificar discrepâncias
    
    2. Gerar atualização
       → Criar/modificar seção relevante
       → Manter formatação consistente
       → Adicionar exemplos se necessário
    
    3. Validar
       → Verificar links internos
       → Conferir código de exemplo
       → Validar formatação markdown
    
    4. Aplicar mudanças
       → Atualizar arquivo(s) de docs
       → Atualizar índice se necessário
       → Gerar changelog entry
  
  output_format:
    type: "diff_report + updated_files"
    files_modified: []
    changes_summary: ""
  
  exemplo_uso:
    "(executar atualizar_documentacao target_module=vqvae change_type=modificado)"
```

**Execução**:
```
User: "(executar atualizar_documentacao target_module=vqvae change_type=modificado)"

Agent:
[STEP 1] Detectando mudanças em core/reasoning/vqvae/model.py...
  → Comparando com docs/modules/03_vqvae.md
  → Detectado: Nova função forward_with_head_mask()

[STEP 2] Gerando atualização...
  → Adicionando seção sobre head ablation
  → Criando exemplo de uso

[STEP 3] Validando...
  ✅ Links internos OK
  ✅ Código testado
  ✅ Markdown válido

[STEP 4] Aplicando mudanças...
  ✅ docs/modules/03_vqvae.md atualizado
  ✅ Adicionado ao CHANGELOG.md

📄 Arquivos modificados:
  - docs/modules/03_vqvae.md (+15 lines)
  - CHANGELOG.md (+3 lines)
```

---

### 3. Workflow: `revisar_modulo`

```yaml
revisar_modulo:
  description: "Revisão técnica completa de um módulo"
  
  parameters:
    - module_name: "nome do módulo"
  
  contexts_required:
    - [module_name] (código do módulo)
    - tests/test_[module_name].py (testes)
    - docs/modules/[module_name].md (docs)
  
  steps:
    1. Análise de código
       → Detectar code smells
       → Verificar type hints
       → Avaliar complexidade
    
    2. Análise de testes
       → Verificar cobertura
       → Identificar casos faltantes
       → Avaliar qualidade dos testes
    
    3. Análise de documentação
       → Verificar se docs refletem código
       → Avaliar clareza
       → Sugerir exemplos adicionais
    
    4. Gerar report de revisão
       → Score geral (0-100)
       → Issues encontradas
       → Sugestões de melhoria
  
  exemplo_uso:
    "(executar revisar_modulo module_name=oracle)"
```

---

### 4. Workflow: `criar_modulo`

```yaml
criar_modulo:
  description: "Scaffold completo de novo módulo"
  
  parameters:
    - module_name: "nome do novo módulo"
    - module_type: "agent|learning|reasoning|memory"
    - description: "breve descrição"
  
  steps:
    1. Criar estrutura de arquivos
       → core/[type]/[module_name].py
       → tests/test_[module_name].py
       → docs/modules/[nn]_[module_name].md
    
    2. Gerar template de código
       → Imports padrão
       → Classe base com docstrings
       → Métodos básicos
    
    3. Gerar template de testes
       → Setup/teardown
       → Testes básicos
    
    4. Gerar documentação inicial
       → Seções padrão
       → Placeholder para exemplos
    
    5. Atualizar índices
       → Adicionar em STRUCTURE.md
       → Adicionar em README.md
       → Criar alias em MAPA_CONTEXTOS.md
  
  exemplo_uso:
    "(executar criar_modulo module_name=reinforcement_agent module_type=agent description='Agente de aprendizado por reforço')"
```

---

### 5. Workflow: `debug_completo`

```yaml
debug_completo:
  description: "Debugging sistemático de um problema"
  
  parameters:
    - problema: "descrição do problema"
    - modulo_afetado: "módulo onde ocorre"
  
  contexts_required:
    - [modulo_afetado]
    - logs/ (se disponível)
    - tests/test_[modulo_afetado].py
  
  steps:
    1. Reproduzir problema
       → Analisar descrição
       → Identificar arquivo/função exata
       → Localizar linha suspeita
    
    2. Análise de causa raiz
       → Examinar lógica
       → Verificar inputs/outputs
       → Checar dependências
    
    3. Propor solução
       → Sugerir fix
       → Mostrar diff
       → Explicar raciocínio
    
    4. Gerar teste regressão
       → Criar teste que falha com bug
       → Verificar que passa com fix
  
  exemplo_uso:
    "(executar debug_completo problema='VQ-VAE retorna NaN' modulo_afetado=vqvae)"
```

---

### 6. Workflow: `onboarding_dev`

```yaml
onboarding_dev:
  description: "Guia completo para novo desenvolvedor"
  
  contexts_required:
    - readme
    - estrutura
    - visao_geral
    - arch_tecnica
  
  steps:
    1. Introdução ao projeto
       → O que é Alexandria
       → Objetivos principais
       → Tecnologias usadas
    
    2. Tour pela estrutura
       → Explicar cada pasta
       → Módulos principais
       → Fluxo de dados
    
    3. Setup inicial
       → Dependências
       → Configuração
       → Primeiro teste
    
    4. Próximos passos
       → Tarefas para iniciantes
       → Recursos úteis
       → Como contribuir
  
  exemplo_uso:
    "(executar onboarding_dev)"
```

---

## 🎛️ Workflows Parametrizados

### Sintaxe com Parâmetros

```
(executar [workflow] param1=valor1 param2=valor2)
```

### Exemplo Real

```
(executar atualizar_documentacao target_module=abduction change_type=novo)
```

**Processamento**:
1. Parse dos parâmetros
2. Validação (target_module existe? change_type válido?)
3. Execução adaptada aos parâmetros

---

## 🔄 Workflows Encadeados

### Sintaxe

```
(pipeline [workflow1] → [workflow2] → [workflow3])
```

### Exemplo: Pipeline de Release

```
(pipeline revisar_modulo module_name=vqvae → atualizar_documentacao target_module=vqvae → criar_changelog)
```

**Execução**:
1. Revisa código do vqvae
2. Atualiza docs baseado na revisão
3. Gera entry no changelog

---

## 📂 Estrutura Final

```
.cosmic_garden/
├── MAPA_CONTEXTOS.md         ← Nível 1 e 2 (aliases e contextos)
├── WORKFLOWS.md              ← Nível 3 (workflows)
├── README.md                 ← Protocolo completo
└── CHANGELOG.md              ← Histórico
```

---

## 🎯 Implementação Sugerida

### Fase 1: Workflows Básicos (Semana 1)

Implementar 3 workflows essenciais:
- `analisar_estrutura`
- `atualizar_documentacao`
- `revisar_modulo`

### Fase 2: Workflows Avançados (Semana 2)

Adicionar:
- `criar_modulo`
- `debug_completo`
- `onboarding_dev`

### Fase 3: Workflows Customizados (Contínuo)

Criar workflows específicos conforme necessidade do projeto.

---

## 💡 Vantagens dos Workflows

✅ **Automação**: Tarefas repetitivas viram comandos simples  
✅ **Consistência**: Sempre segue o mesmo processo  
✅ **Documentação viva**: Workflows documentam procedimentos  
✅ **Escalabilidade**: Fácil adicionar novos workflows  
✅ **Reprodutibilidade**: Mesmo resultado toda vez  

---

## 🔮 Exemplo de Uso Completo

```
[SESSÃO DE TRABALHO: Adicionar novo módulo]

User: "(executar criar_modulo module_name=reward_predictor module_type=learning description='Preditor de recompensas para RL')"

Agent:
[STEP 1/5] Criando estrutura...
  ✅ core/learning/reward_predictor.py
  ✅ tests/test_reward_predictor.py
  ✅ docs/modules/10_reward_predictor.md

[STEP 2/5] Gerando código template...
  ✅ Classe RewardPredictor criada
  ✅ Métodos predict(), train(), evaluate() adicionados
  ✅ Docstrings completas

[STEP 3/5] Gerando testes...
  ✅ test_predict_basic()
  ✅ test_train_updates_weights()
  ✅ test_evaluate_returns_metrics()

[STEP 4/5] Gerando documentação...
  ✅ Seções: Overview, Architecture, Usage, Integration
  ✅ Placeholders para exemplos

[STEP 5/5] Atualizando índices...
  ✅ STRUCTURE.md
  ✅ README.md (seção de módulos)
  ✅ MAPA_CONTEXTOS.md (alias: reward_predictor)

📦 Módulo 'reward_predictor' criado!
📝 Arquivos: 3 criados, 3 atualizados

---

User: "Ótimo! Agora me explique a arquitetura + (ativar reward_predictor)"

Agent: [Lê core/learning/reward_predictor.py recém-criado]
       "O RewardPredictor foi criado com arquitetura modular..."
```

---

**Status**: Conceito completo  
**Próximo passo**: Criar `WORKFLOWS.md` no Alexandria
