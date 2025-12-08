---
description: Review completo de código antes de merge (análise + review + docs)
---

# Workflow: Review Completo para Merge

Execute antes de fazer merge de branches ou commits importantes.

## Passo 1: Identificar Arquivos Modificados
Liste todos os arquivos `.py` modificados (se possível, use `git diff --name-only`).

## Passo 2: Code Review de Cada Arquivo
// turbo
Para cada arquivo modificado, execute `.prompts/code_review.md`:
```yaml
alvo: "[arquivo modificado]"
foco: "todos"
```

## Passo 3: Verificar Impacto nas Dependências
Analise quais módulos dependem dos arquivos modificados e verifique se há breaking changes.

## Passo 4: Atualizar Documentação
// turbo
Execute `.prompts/atualizar_documentacao.md` para sincronizar docs com as mudanças.

## Passo 5: Relatório de Review
Gere relatório consolidado em `docs/reports/review_YYYY-MM-DD.md`:
- Score geral do review
- Issues críticas (se houver)
- Aprovação ou bloqueios
