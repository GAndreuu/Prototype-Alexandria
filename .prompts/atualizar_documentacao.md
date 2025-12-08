# 🔄 PROMPT: Atualizar Documentação Existente

---

## INSTRUÇÃO PARA O AGENTE

Execute uma sincronização completa entre o código atual e a documentação existente, identificando discrepâncias e atualizando os documentos.

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Auditoria de Código
1. Varra toda a estrutura de arquivos atual
2. Liste todas as funções, classes e módulos públicos
3. Identifique mudanças recentes (novos arquivos, modificações)
4. Mapeie dependências atuais

### FASE 2: Auditoria de Documentação
1. Localize todos os arquivos .md existentes
2. Leia cada documento de documentação
3. Extraia o que está documentado na pasta doc atualmente
4. Crie uma lista do "estado documentado"

### FASE 3: Análise de Discrepâncias

Compare e identifique:

| Tipo | Código | Documentação | Status |
|------|--------|--------------|--------|
| Função/Classe X | ✅ Existe | ❌ Não documentada | **FALTANDO** |
| Função/Classe Y | ❌ Removida | ✅ Documentada | **OBSOLETA** |
| Função/Classe Z | ✅ Modificada | ⚠️ Desatualizada | **ATUALIZAR** |
| Função/Classe W | ✅ Existe | ✅ Documentada | OK |

### FASE 4: Geração de Atualizações

Para cada discrepância, gere:

```markdown
## 📝 Atualização: [NOME_DO_ARQUIVO]

### Alteração
[Descrição do que mudou]

### Antes (conteúdo atual)
```
[código/texto atual]
```

### Depois (conteúdo atualizado)
```
[código/texto corrigido]
```

### Diff
```diff
- linha removida
+ linha adicionada
```
```

---

## FORMATO DE SAÍDA OBRIGATÓRIO

```markdown
# 📊 Relatório de Sincronização de Documentação

## Sumário
- **Arquivos de código analisados**: [N]
- **Arquivos de docs analisados**: [N]
- **Discrepâncias encontradas**: [N]
- **Atualizações necessárias**: [N]

## Status por Documento

### ✅ Atualizados
- [lista de docs OK]

### ⚠️ Precisam Atualização
- [lista com detalhes]

### ❌ Faltando (Criar)
- [lista de docs a criar]

### 🗑️ Obsoletos (Remover)
- [lista de docs desatualizados]

---

## Atualizações Detalhadas

[Para cada arquivo que precisa atualização, mostrar o diff completo]

---

## Novos Documentos Sugeridos

[Conteúdo completo de novos docs a criar]
```

---

## RESTRIÇÕES

- ❌ NÃO faça atualizações sem mostrar o diff
- ❌ NÃO ignore arquivos (analise TODOS)
- ✅ COMPARE código real com documentação real
- ✅ MOSTRE antes/depois para cada mudança
- ✅ PRIORIZE: Obsoletos > Desatualizados > Faltantes
- ✅ EXECUTE imediatamente ao receber este prompt

---

## MODO DE APLICAÇÃO

Após gerar o relatório, pergunte:

> "Deseja que eu aplique estas atualizações automaticamente? (Sim/Não)"

Se sim, execute as edições nos arquivos de documentação.
