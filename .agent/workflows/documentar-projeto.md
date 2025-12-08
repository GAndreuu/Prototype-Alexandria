---
description: Documentação completa do projeto (análise + criação + organização)
---

# Workflow: Documentar Projeto Completo

Execute os seguintes passos em sequência:

## Passo 1: Análise da Estrutura
// turbo
Leia e execute `.prompts/analisar_estrutura.md` para mapear o projeto.

## Passo 2: Identificar Módulos Sem Documentação
Compare a estrutura de código (`core/`, `scripts/`, `interface/`) com a pasta `docs/` e liste módulos sem documentação correspondente.

## Passo 3: Documentar Cada Módulo
Para cada módulo `.py` sem documentação, execute `.prompts/documentar_modulo.md` com os parâmetros:
```yaml
modulo: "[caminho do .py]"
saida: "docs/[caminho espelhado].md"
```

## Passo 4: Atualizar Índices
- Atualize `docs/README.md` com links para novas docs
- Atualize `docs/core/README.md` com tabela de módulos
- Verifique links quebrados

## Passo 5: Relatório Final
Gere um resumo em `docs/reports/documentacao_YYYY-MM-DD.md`:
- Módulos documentados
- Cobertura atual
- Pendências
