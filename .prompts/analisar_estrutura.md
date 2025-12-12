# 🔍 PROMPT: Analisar Estrutura Completa

---

## INSTRUÇÃO PARA O AGENTE

Execute uma análise estrutural completa deste projeto seguindo o protocolo abaixo.

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Varredura Topológica
1. Liste TODAS as pastas e arquivos do projeto recursivamente
2. Identifique a profundidade de cada nível (root = 0)
3. Classifique cada item: `[DIR]`, `[FILE]`, `[CONFIG]`, `[DOC]`, `[TEST]`

### FASE 2: Mapeamento de Módulos
Para cada pasta principal, identifique:
- **Propósito**: O que este módulo faz?
- **Dependências**: De quais outros módulos depende?
- **Arquivos-chave**: Quais são os arquivos mais importantes?
- **Entry points**: Onde está o ponto de entrada?

### FASE 3: Detecção de Padrões
Analise e reporte:
- [ ] Padrão arquitetural (MVC, Clean Architecture, Modular, etc.)
- [ ] Convenções de nomenclatura usadas
- [ ] Estrutura de testes (se existir)
- [ ] Configurações e variáveis de ambiente

### FASE 4: Avaliação de Qualidade
Verifique:
- [ ] Arquivos órfãos (sem uso aparente)
- [ ] Pastas vazias
- [ ] Documentação ausente
- [ ] Inconsistências na estrutura

### FASE 5: Reality Check (Capabilities vs Concepts)
CRÍTICO: Compare o código encontrado em `core/` com `docs/concepts/`.
- [ ] O conceito descrito em `docs/concepts` realmente existe no código?
- [ ] Existe código complexo sem conceito correspondente?
- [ ] Classifique cada conceito como:
    - ✅ **Realidade**: Código implementa fielmente o conceito.
    - ⚠️ **Aspiração**: Conceito documentado mas implementação é mockup/stub.
    - ❌ **Alucinação**: Conceito existe apenas na doc, zero código.

---

## FORMATO DE SAÍDA OBRIGATÓRIO

```markdown
# 📊 Relatório de Análise Estrutural

## Resumo Executivo
- **Total de pastas**: [N]
- **Total de arquivos**: [N]
- **Profundidade máxima**: [N] níveis
- **Padrão arquitetural**: [identificado]

## Árvore de Diretórios
[Representação visual completa]

## Módulos Principais
### 1. [nome_modulo]
- **Caminho**: /path/to/module
- **Propósito**: [descrição]
- **Arquivos**: [lista]
- **Dependências**: [lista]

## Pontos de Atenção
⚠️ [Lista de problemas encontrados]

## 🧠 Reality Check (Conceito vs Código)
| Conceito | Status | Evidência no Código |
|----------|--------|---------------------|
| [Nome] | ✅/⚠️/❌ | `path/to/evidence.py` |


## Recomendações
1. [Sugestão de melhoria]
2. [Sugestão de melhoria]
```

---

## RESTRIÇÕES

- ❌ NÃO resuma de forma genérica
- ❌ NÃO omita arquivos ou pastas
- ✅ SEJA específico e literal
- ✅ CITE caminhos completos
- ✅ EXECUTE imediatamente ao receber este prompt
