---
description: Debug profundo com análise de dependências e fluxo de dados
---

# Workflow: Debug Profundo

Parâmetros:
```yaml
erro: "[MENSAGEM DE ERRO OU DESCRIÇÃO DO PROBLEMA]"
arquivo: "[ARQUIVO ONDE O ERRO OCORRE - opcional]"
```

---

## Passo 1: Análise Inicial
// turbo
Execute `.prompts/debug_completo.md` com o erro.

## Passo 2: Rastrear Dependências
Se o debug inicial não resolver:
1. Identifique todos os imports do arquivo problemático
2. Execute análise em cada dependência
3. Mapeie o fluxo de dados até encontrar a origem

## Passo 3: Análise de Estado
Verifique:
- Valores de variáveis em pontos críticos
- Estado do sistema no momento do erro
- Logs relacionados

## Passo 4: Propor Correção
Apresente:
- Root cause identificada
- Correção sugerida (com diff)
- Teste para validar a correção

## Passo 5: Aplicar e Validar
// turbo
1. Aplique a correção
2. Execute testes relacionados
3. Confirme que o erro foi resolvido

## Passo 6: Documentar Bug
Adicione entrada em `docs/reports/bugs/` com:
- Descrição do bug
- Root cause
- Solução aplicada
- Prevenção futura
