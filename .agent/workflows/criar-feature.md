---
description: Criar nova feature completa (módulo + docs + testes)
---

# Workflow: Criar Nova Feature

Parâmetros:
```yaml
nome: "[NOME_DA_FEATURE]"
descricao: "[DESCRIÇÃO BREVE]"
categoria: "[agents|learning|memory|reasoning|topology|utils]"
```

---

## Passo 1: Criar Estrutura do Módulo
// turbo
Execute `.prompts/criar_modulo.md` com os parâmetros acima.

## Passo 2: Implementar Código Base
Crie o arquivo `core/{categoria}/{nome}.py` com:
- Classe principal
- Dataclasses de entrada/saída
- Métodos públicos com docstrings
- Type hints completos

## Passo 3: Criar Testes
// turbo
Crie `tests/test_{nome}.py` com:
- Testes unitários para cada método público
- Mocks para dependências externas
- Edge cases

## Passo 4: Documentar
// turbo
Execute `.prompts/documentar_modulo.md`:
```yaml
modulo: "core/{categoria}/{nome}.py"
saida: "docs/core/{categoria}/{nome}.md"
```

## Passo 5: Atualizar Índices
- Adicione entrada em `docs/core/{categoria}/README.md`
- Adicione entrada em `docs/core/README.md`
- Atualize diagrama de dependências se necessário

## Passo 6: Verificação Final
// turbo
Execute `.prompts/code_review.md` no novo módulo para validar qualidade.
