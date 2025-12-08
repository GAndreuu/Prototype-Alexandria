# ⚡ PROMPT: Revisão de Código (Code Review)

---

## INSTRUÇÃO PARA O AGENTE

Execute uma revisão técnica completa do módulo ou arquivo especificado.

---

## PARÂMETROS (preencha antes de usar)

```yaml
alvo: "[CAMINHO DO ARQUIVO OU MÓDULO]"
foco: "[segurança|performance|qualidade|todos]"
```

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Análise de Código

Leia o código e analise:

#### Qualidade Geral
- [ ] Nomenclatura clara e consistente
- [ ] Funções com responsabilidade única
- [ ] Complexidade ciclomática aceitável
- [ ] DRY (Don't Repeat Yourself)
- [ ] Tratamento de erros adequado

#### Type Safety
- [ ] Type hints presentes
- [ ] Types consistentes
- [ ] Nullable handling

#### Documentação
- [ ] Docstrings presentes
- [ ] Comentários úteis (não óbvios)
- [ ] README/docs atualizados

### FASE 2: Análise de Segurança

- [ ] Input validation
- [ ] SQL injection risks
- [ ] XSS vulnerabilities
- [ ] Secrets hardcoded
- [ ] Permissions adequadas

### FASE 3: Análise de Performance

- [ ] Loops eficientes
- [ ] Queries otimizadas
- [ ] Cache quando apropriado
- [ ] Memory leaks potenciais
- [ ] Async quando benéfico

### FASE 4: Análise de Testes

- [ ] Cobertura adequada
- [ ] Edge cases testados
- [ ] Mocks apropriados
- [ ] Tests independentes

---

## FORMATO DE SAÍDA OBRIGATÓRIO

```markdown
# 📋 Code Review: [nome_do_arquivo]

## Score Geral: [X]/100

### Breakdown
| Categoria | Score | Status |
|-----------|-------|--------|
| Qualidade | [X]/25 | 🟢/🟡/🔴 |
| Segurança | [X]/25 | 🟢/🟡/🔴 |
| Performance | [X]/25 | 🟢/🟡/🔴 |
| Testes | [X]/25 | 🟢/🟡/🔴 |

---

## Issues Encontradas

### 🔴 Críticas (bloqueia merge)
1. **[Título]** - Linha [N]
   - Problema: [descrição]
   - Fix sugerido:
   ```diff
   - código atual
   + código corrigido
   ```

### 🟡 Importantes (deve corrigir)
[lista]

### 🟢 Sugestões (nice to have)
[lista]

---

## Pontos Positivos
✅ [O que está bom no código]

## Recomendações Finais
1. [Ação prioritária]
2. [Ação secundária]
```

---

## RESTRIÇÕES

- ❌ NÃO seja genérico ("melhorar nomenclatura")
- ✅ CITE linhas específicas
- ✅ MOSTRE diffs para cada sugestão
- ✅ PRIORIZE por severidade
- ✅ EXECUTE imediatamente ao receber este prompt
