# 🚀 Prompts & Workflows - Alexandria

**Uso**: Cole prompts no chat OU ative workflows via slash commands.

---

## 📋 Prompts Disponíveis

### Documentação & Análise

| # | Prompt | Descrição | Quando Usar |
|---|--------|-----------|-------------|
| 1 | [analisar_estrutura.md](./analisar_estrutura.md) | Análise completa da estrutura | Entender projeto novo |
| 2 | [criar_documentacao.md](./criar_documentacao.md) | Criar docs do zero | Projeto sem docs |
| 3 | [atualizar_documentacao.md](./atualizar_documentacao.md) | Sincronizar docs com código | Docs desatualizadas |
| 4 | [documentar_modulo.md](./documentar_modulo.md) | Documentar módulo específico | Doc individual |

### Desenvolvimento

| # | Prompt | Descrição | Quando Usar |
|---|--------|-----------|-------------|
| 5 | [criar_modulo.md](./criar_modulo.md) | Scaffold de novo módulo | Adicionar feature |
| 6 | [debug_completo.md](./debug_completo.md) | Debugging sistemático | Bug complexo |
| 7 | [code_review.md](./code_review.md) | Revisão técnica completa | Antes de merge |

---

## ⚡ Workflows (Slash Commands)

Workflows encadeiam múltiplos prompts automaticamente:

| Comando | Descrição | Prompts Usados |
|---------|-----------|----------------|
| `/documentar-projeto` | Documentação completa | analisar + documentar_modulo |
| `/review-completo` | Review antes de merge | code_review + atualizar_documentacao |
| `/criar-feature` | Nova feature completa | criar_modulo + documentar + code_review |
| `/debug-profundo` | Debug com rastreamento | debug_completo + análise de deps |

**Como usar**: Digite `/documentar-projeto` no chat e o agente executa todos os passos!

---

## 🎯 Como Usar Prompts

```
1. Abra o arquivo .md do prompt desejado
2. Copie TODO o conteúdo (Ctrl+A → Ctrl+C)
3. Cole no chat da IDE (Ctrl+V)
4. ✨ O agente executa automaticamente
```

---

## 📁 Estrutura

```
.prompts/                          ← Prompts individuais
├── README.md                      ← Você está aqui
├── analisar_estrutura.md
├── criar_documentacao.md
├── atualizar_documentacao.md
├── documentar_modulo.md           ← NOVO
├── criar_modulo.md
├── debug_completo.md
└── code_review.md

.agent/workflows/                  ← Workflows (slash commands)
├── documentar-projeto.md
├── review-completo.md
├── criar-feature.md
└── debug-profundo.md
```
