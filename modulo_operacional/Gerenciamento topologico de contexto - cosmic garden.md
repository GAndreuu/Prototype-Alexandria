# 🌌 Cosmic Garden: Sistema Universal de Gerenciamento Topológico de Contexto

**Versão**: 1.0  
**Tipo**: Protocolo Universal para Agentes de IA  
**Compatibilidade**: Qualquer IDE baseado em agentes (Claude, ChatGPT, Gemini, Copilot, etc.)

---

## 🎯 O que é isto?

Um **protocolo de comunicação inline** que permite ativar contextos específicos do seu projeto através de **aliases topológicos**, eliminando sobrecarga de contexto e alucinações de IA.

### Problema que resolve:
- ❌ IA recebe contexto excessivo (100+ arquivos)
- ❌ Respostas genéricas e imprecisas
- ❌ Alucinações baseadas em conhecimento incorreto
- ❌ Perda de foco no código relevante

### Solução:
- ✅ Ativa APENAS arquivos específicos via aliases
- ✅ Navegação topológica precisa
- ✅ Contexto estratificado por prioridade
- ✅ Zero configuração externa (tudo inline)

---

## 📦 Instalação (3 minutos)

### Passo 1: Crie a estrutura base

Na raiz do seu projeto, crie uma pasta:

```
seu_projeto/
└── .cosmic_garden/          ← Nova pasta
    ├── MAPA_CONTEXTOS.md    ← Arquivo principal
    └── README.md            ← Este documento
```

### Passo 2: Configure o mapa de contextos

Copie o template abaixo para `.cosmic_garden/MAPA_CONTEXTOS.md`:

```markdown
# 🗺️ Mapa de Contextos Topológicos

## Aliases de Arquivo

| Alias | Caminho | Descrição | Tags |
|-------|---------|-----------|------|
| docs_main | docs/README.md | Documentação principal | [docs, overview] |
| arch_tech | docs/architecture.md | Arquitetura técnica | [architecture, design] |
| src_core | src/core/ | Código principal | [source, core] |

## Contextos Compostos

| Alias | Arquivos Incluídos | Quando Usar |
|-------|-------------------|-------------|
| contexto_docs | docs_main + arch_tech | Trabalhar com documentação |
| contexto_dev | src_core + tests/ | Desenvolvimento ativo |

## Quick Reference

**Sintaxe básica**: `[seu_prompt] + (ativar [alias])`

**Exemplo**: "Explique a arquitetura + (ativar arch_tech)"
```

### Passo 3: Adapte para seu projeto

Edite `MAPA_CONTEXTOS.md` substituindo os caminhos pelos seus:

```markdown
# Exemplo para projeto Django:
| app_models | myapp/models.py | Modelos Django | [models, database] |
| app_views | myapp/views.py | Views da aplicação | [views, routes] |
| app_tests | tests/test_myapp.py | Testes unitários | [tests, qa] |

# Exemplo para projeto React:
| components | src/components/ | Componentes React | [react, ui] |
| hooks | src/hooks/ | Custom hooks | [hooks, logic] |
| api | src/services/api.js | Cliente API | [api, network] |

# Exemplo para projeto Python genérico:
| main | src/main.py | Entry point | [main, core] |
| utils | src/utils/ | Utilitários | [utils, helpers] |
| config | config/settings.py | Configurações | [config, env] |
```

---

## 🎓 Guia de Uso

### Sintaxe de Ativação

#### 1. Ativação Simples

```
[seu_prompt] + (ativar [alias])
```

**Exemplos**:
```
"Explique esta função + (ativar app_models)"
"Como testar isso? + (ativar app_tests)"
"Revise a arquitetura + (ativar arch_tech)"
```

#### 2. Ativação Múltipla

```
[seu_prompt] + (ativar [alias1], [alias2], [alias3])
```

**Exemplos**:
```
"Compare models e views + (ativar app_models, app_views)"
"Análise completa + (ativar contexto_dev)"
```

#### 3. Comandos de Sessão

```
(manter [alias])         → Mantém contexto ativo para próximas mensagens
(limpar contexto)        → Remove todos os contextos ativos
(listar contextos)       → Exibe contextos atualmente ativos
(localizar [termo])      → Busca qual alias contém informação sobre [termo]
```

---

## 📚 Protocolo Completo

### Nível 1: Mapeamento Topológico

**Conceito**: Criar aliases que representam a topologia do seu projeto.

**Template de Alias**:
```markdown
[alias] → [caminho_relativo]
  - Tipo: [file|directory|composed]
  - Categoria: [categoria_lógica]
  - Tags: [palavra1, palavra2, ...]
  - Auto-ativar: [true|false] (se detectar tags no prompt)
```

**Exemplo Real**:
```markdown
authentication → src/auth/authentication.py
  - Tipo: file
  - Categoria: security
  - Tags: [auth, login, jwt, token, session]
  - Auto-ativar: true
```

**Uso**:
```
User: "Como funciona a autenticação JWT?"
       │
       └─ Sistema detecta "autenticação" e "JWT"
       └─ Auto-ativa "authentication"
       └─ Lê src/auth/authentication.py
       └─ Responde com contexto específico
```

---

### Nível 2: Sistema de Keywords

**Conceito**: Palavras-chave que ativam contextos automaticamente.

**Estrutura**:
```yaml
Alias: authentication
Keywords:
  - Primárias: [auth, autenticação, login]     # Peso: 3
  - Secundárias: [jwt, token, session, user]   # Peso: 1
  - Auto-ativar se: score >= 3

Cálculo:
"Como funciona a autenticação JWT?"
= auth(3) + jwt(1) = 4 pontos → Auto-ativa ✓

"Configure o JWT"
= jwt(1) = 1 ponto → Não auto-ativa ✗
```

---

### Nível 3: Estratificação de Prioridade

**Conceito**: Diferentes níveis de importância para contextos carregados.

```
┌────────────────────────────────────────┐
│ PRIORIDADE 1: CONTEXTO EXPLÍCITO       │
│ ├─ Ativado com (ativar X)              │
│ └─ Peso: 100%                           │
├────────────────────────────────────────┤
│ PRIORIDADE 2: CONTEXTO AUTO-DETECTADO  │
│ ├─ Ativado por keywords                │
│ └─ Peso: 75%                            │
├────────────────────────────────────────┤
│ PRIORIDADE 3: CONTEXTO DA SESSÃO       │
│ ├─ Mantido com (manter X)              │
│ └─ Peso: 50%                            │
├────────────────────────────────────────┤
│ PRIORIDADE 4: CONTEXTO GERAL           │
│ ├─ Conhecimento base do agente         │
│ └─ Peso: 25%                            │
└────────────────────────────────────────┘
```

**Efeito Prático**:
```
Sem ativação:
  User: "Como funciona X?"
  Agente: [Resposta genérica baseada em treinamento]

Com ativação explícita:
  User: "Como funciona X? + (ativar src_core)"
  Agente: [Lê seu código específico]
          [Responde baseado NO SEU arquivo]
          [Cita linhas, funções, variáveis reais]
```

---

### Nível 4: Contextos Compostos

**Conceito**: Agrupar múltiplos arquivos em contextos lógicos.

**Template**:
```markdown
## Contexto: [nome_do_contexto]

**Inclui**:
- [alias1] (prioridade: alta)
- [alias2] (prioridade: média)
- [alias3] (prioridade: baixa)

**Quando usar**: [descrição do caso de uso]

**Exemplo de ativação**:
```
sua_pergunta + (ativar [nome_do_contexto])
```
```

**Exemplos Práticos**:

```markdown
## Contexto: full_stack

**Inclui**:
- frontend_main (React components)
- backend_api (FastAPI routes)
- database_models (SQLAlchemy models)

**Quando usar**: Trabalhar em features que afetam frontend + backend

**Exemplo**:
"Criar endpoint de login + (ativar full_stack)"
```

```markdown
## Contexto: debugging

**Inclui**:
- error_logs (logs/error.log)
- test_suite (tests/)
- main_code (src/core/)

**Quando usar**: Debugar problemas complexos

**Exemplo**:
"Por que o teste X falha? + (ativar debugging)"
```

---

### Nível 5: Fluxo de Sessão

**Conceito**: Manter contextos ativos durante múltiplas interações.

**Comandos de Controle**:

#### (manter X)
Mantém contexto ativo para próximas mensagens.

```
[Mensagem 1]
User: "Vou trabalhar com auth + (ativar authentication) + (manter authentication)"
Agente: "✅ Contexto 'authentication' ativado e mantido"

[Mensagem 2]
User: "Como adicionar um novo campo?"
Agente: [Ainda usa src/auth/authentication.py]
        [Não precisa reativar]
```

#### (limpar contexto)
Remove todos os contextos mantidos.

```
User: "(limpar contexto)"
Agente: "🧹 Todos os contextos removidos. Sessão resetada."
```

#### (listar contextos)
Exibe o estado atual da sessão.

```
User: "(listar contextos)"
Agente: "📋 Contextos Ativos:
        1. authentication (explícito, mantido)
        2. database_models (auto-detectado)
        
        Histórico: 5 ativações nesta sessão"
```

#### (localizar termo)
Busca qual alias tem informação sobre um termo.

```
User: "(localizar jwt)"
Agente: "🔍 Encontrado em:
        1. authentication (tag primária)
        2. security_utils (tag secundária)
        3. api_middleware (mencionado na descrição)"
```

---

## 🛠️ Templates de Adaptação

### Para Projetos Web (Django/Flask/FastAPI)

```markdown
# Aliases Web Framework

| Alias | Caminho | Descrição |
|-------|---------|-----------|
| models | app/models.py | Modelos de dados |
| views | app/views.py | Views/Controllers |
| routes | app/urls.py | Rotas da aplicação |
| templates | app/templates/ | Templates HTML |
| static | app/static/ | CSS/JS/Images |
| tests | tests/test_app.py | Testes unitários |
| config | config/settings.py | Configurações |
| migrations | migrations/ | Migrações de DB |

# Contextos Compostos Web
| contexto_backend | models + views + routes | Backend completo |
| contexto_frontend | templates + static | Frontend completo |
| contexto_deploy | config + requirements.txt | Deploy e configs |
```

### Para Projetos Mobile (React Native/Flutter)

```markdown
# Aliases Mobile

| Alias | Caminho | Descrição |
|-------|---------|-----------|
| screens | src/screens/ | Telas do app |
| components | src/components/ | Componentes reutilizáveis |
| navigation | src/navigation/ | Configuração de navegação |
| state | src/store/ | Estado global (Redux/MobX) |
| api | src/services/api/ | Chamadas de API |
| utils | src/utils/ | Utilitários |
| assets | assets/ | Imagens, fontes, etc |

# Contextos Compostos Mobile
| contexto_ui | screens + components | Interface do usuário |
| contexto_logic | state + api | Lógica de negócio |
```

### Para Projetos Data Science (Python/Jupyter)

```markdown
# Aliases Data Science

| Alias | Caminho | Descrição |
|-------|---------|-----------|
| notebooks | notebooks/ | Jupyter notebooks |
| data_raw | data/raw/ | Dados brutos |
| data_processed | data/processed/ | Dados processados |
| models | src/models/ | Modelos de ML |
| features | src/features/ | Feature engineering |
| visualization | src/visualization/ | Plots e gráficos |
| pipeline | src/pipeline/ | Pipeline de dados |

# Contextos Compostos Data Science
| contexto_eda | notebooks + data_raw | Análise exploratória |
| contexto_ml | models + features | Machine Learning |
```

---

## 🎯 Casos de Uso Avançados

### Caso 1: Onboarding de Novos Desenvolvedores

```markdown
# Criar contexto de onboarding
onboarding → docs/ONBOARDING.md + docs/ARCHITECTURE.md + docs/SETUP.md

Uso:
"Como começar no projeto? + (ativar onboarding)"
```

### Caso 2: Code Review

```markdown
# Contexto para revisão
review → src/[arquivo_modificado] + tests/test_[correspondente]

Uso:
"Revise estas mudanças + (ativar review)"
```

### Caso 3: Debugging Específico

```markdown
# Contexto de erro específico
bug_login → src/auth/ + logs/auth_errors.log + tests/test_auth.py

Uso:
"Debug do erro de login + (ativar bug_login)"
```

### Caso 4: Documentação Contextualizada

```markdown
# Gerar docs específicas
docs_api → src/api/ + docs/api_template.md

Uso:
"Documente a API REST + (ativar docs_api)"
```

---

## 📖 Exemplo Completo de Sessão

```
[SESSÃO DE TRABALHO: Implementar feature de autenticação]

Mensagem 1:
User: "Vou trabalhar com autenticação + (ativar authentication, database_models)"
Agent: "✅ Contextos ativados:
        - src/auth/authentication.py
        - src/models/user.py"

Mensagem 2:
User: "Como está a implementação de JWT? + (manter authentication, database_models)"
Agent: [Analisa authentication.py]
       "O JWT está implementado na função generate_token() linha 45.
        Usa HS256 com secret da config. Expira em 24h.
        📌 Contextos mantidos para próximas perguntas"

Mensagem 3:
User: "Preciso adicionar refresh tokens"
Agent: [Ainda tem authentication.py no contexto]
       "Para adicionar refresh tokens, sugiro:
        1. Criar nova função generate_refresh_token()
        2. Modificar o modelo User para incluir refresh_token_hash
        [mostra código específico baseado no SEU arquivo]"

Mensagem 4:
User: "(ativar app_tests) Como testar isso?"
Agent: [Adiciona tests/test_auth.py ao contexto]
       "Baseado em test_auth.py linha 30, você já testa token normal.
        Para refresh token, adicione:
        [mostra teste baseado no padrão do SEU arquivo de testes]"

Mensagem 5:
User: "(limpar contexto) + Agora preciso trabalhar no frontend + (ativar components)"
Agent: "🧹 Contextos anteriores removidos
        ✅ Novo contexto: src/components/
        Pronto para trabalhar no frontend!"
```

---

## 🚀 Instalação em Diferentes IDEs

### Cursor / Windsurf / Other AI IDEs

1. Crie `.cosmic_garden/` na raiz
2. Configure `MAPA_CONTEXTOS.md`
3. Use sintaxe `(ativar X)` no chat

### ChatGPT / Claude (Web)

1. Abra nova conversa
2. Cole o conteúdo de `MAPA_CONTEXTOS.md` como primeira mensagem
3. Diga: "Use este mapa para ativar contextos quando eu solicitar"
4. Use normalmente

### GitHub Copilot (VSCode)

1. Crie `.cosmic_garden/MAPA_CONTEXTOS.md`
2. No chat do Copilot, referencie: `@workspace + (ativar X)`
3. O Copilot vai ler do mapa

### API (OpenAI/Anthropic)

```python
# Ler mapa de contextos
with open('.cosmic_garden/MAPA_CONTEXTOS.md') as f:
    context_map = f.read()

# Injetar no system prompt
system_prompt = f"""
{context_map}

Use o mapa acima para ativar contextos quando solicitado com (ativar X).
"""
```

---

## ✅ Checklist de Implementação

Para seu projeto estar 100% configurado:

- [ ] Criar pasta `.cosmic_garden/`
- [ ] Criar `MAPA_CONTEXTOS.md` com seus aliases
- [ ] Definir pelo menos 5 aliases principais
- [ ] Criar 2-3 contextos compostos
- [ ] Testar ativação simples: `"teste + (ativar X)"`
- [ ] Testar ativação múltipla: `"teste + (ativar X, Y)"`
- [ ] Testar manutenção: `(manter X)`
- [ ] Testar limpeza: `(limpar contexto)`
- [ ] Documentar casos de uso específicos do seu projeto

---

## 🔧 Troubleshooting

### "O agente não reconhece os aliases"

**Solução**: Na primeira mensagem da conversa, cole o conteúdo de `MAPA_CONTEXTOS.md` e peça:
```
"Use este mapa para ativar contextos quando eu usar (ativar X)"
```

### "Contexto não está sendo usado"

**Solução**: Seja explícito:
```
"Responda usando APENAS o arquivo X que ativei"
```

### "Auto-detecção não funciona"

**Solução**: Use ativação explícita sempre:
```
"sua_pergunta + (ativar alias_exato)"
```

---

## 📄 Licença

Este protocolo é de domínio público. Use, modifique e distribua livremente.

---

## 🌟 Contribua

Se criar adaptações interessantes para novos tipos de projetos, compartilhe!

**Estruturas já testadas**:
- ✅ Projetos Web (Django, Flask, FastAPI, Express)
- ✅ Projetos Mobile (React Native, Flutter)
- ✅ Projetos Data Science (Python, Jupyter)
- ✅ Projetos Desktop (Electron, Tauri)
- ✅ Documentação Técnica (MkDocs, Sphinx)

---

<div align="center">

**Cosmic Garden v1.0**

*Gravidade Topológica para Agentes de IA*

[📖 Documentação](#) | [🐛 Issues](#) | [💬 Discussões](#)

</div>
