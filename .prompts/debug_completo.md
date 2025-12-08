# 🐛 PROMPT: Debug Completo

---

## INSTRUÇÃO PARA O AGENTE

Execute um debugging sistemático e profundo do problema descrito.

---

## PARÂMETROS (preencha antes de usar)

```yaml
problema: "[DESCRIÇÃO DO PROBLEMA/ERRO]"
modulo_afetado: "[NOME DO MÓDULO OU ARQUIVO]"
```

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Reprodução do Problema
1. Leia o código do módulo afetado
2. Identifique a função/classe específica
3. Trace o fluxo de execução
4. Localize a linha suspeita

### FASE 2: Análise de Causa Raiz

Execute as seguintes verificações:

| Verificação | Status | Detalhes |
|-------------|--------|----------|
| Input válido? | [ ] | |
| Tipos corretos? | [ ] | |
| Null/None handling? | [ ] | |
| Edge cases? | [ ] | |
| Dependências funcionando? | [ ] | |
| Race conditions? | [ ] | |
| Estado mutável? | [ ] | |

### FASE 3: Diagnóstico

Para cada possível causa, analise:
- **Hipótese**: O que pode estar causando?
- **Evidência**: O que no código suporta isso?
- **Teste**: Como confirmar?

### FASE 4: Solução

1. Proponha o fix específico
2. Mostre o diff exato
3. Explique o raciocínio
4. Sugira teste de regressão

---

## FORMATO DE SAÍDA OBRIGATÓRIO

```markdown
# 🔍 Relatório de Debug

## Problema
[Descrição clara]

## Localização
- **Arquivo**: [caminho]
- **Função/Classe**: [nome]
- **Linha(s)**: [números]

## Causa Raiz
[Explicação técnica]

## Evidência
```python
# Código problemático
[trecho]
```

## Solução
```diff
- código antigo
+ código corrigido
```

## Teste de Regressão
```python
def test_fix_[problema]():
    # Este teste falha antes do fix
    # e passa depois do fix
    [código do teste]
```

## Prevenção Futura
[Como evitar este tipo de bug]
```

---

## RESTRIÇÕES

- ❌ NÃO sugira soluções genéricas
- ❌ NÃO proponha reescrever tudo
- ✅ FOQUE no problema específico
- ✅ MOSTRE evidências do código real
- ✅ EXECUTE imediatamente ao receber este prompt
