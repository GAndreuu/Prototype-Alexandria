# 🏗️ PROMPT: Criar Novo Módulo

---

## INSTRUÇÃO PARA O AGENTE

Crie um novo módulo completo seguindo os padrões do projeto, incluindo código, testes e documentação.

---

## PARÂMETROS (preencha antes de usar)

```yaml
nome_modulo: "[NOME_DO_MODULO]"
tipo: "[agent|learning|reasoning|memory|utils]"
descricao: "[DESCRIÇÃO BREVE]"
```

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Análise de Padrões
1. Identifique a estrutura de módulos existente
2. Analise convenções de nomenclatura usadas
3. Verifique templates de código existentes
4. Mapeie dependências comuns

### FASE 2: Scaffold do Módulo

Crie os seguintes arquivos:

#### 📄 Código Principal
```
[pasta_do_tipo]/[nome_modulo].py
```
- Imports padrão do projeto
- Classe principal com docstrings
- Métodos básicos (init, process, etc.)
- Type hints completos

#### 📄 Testes
```
tests/test_[nome_modulo].py
```
- Setup/teardown
- Testes unitários básicos
- Mocks quando necessário

#### 📄 Documentação
```
docs/modules/[nome_modulo].md
```
- Visão geral
- API Reference
- Exemplos de uso
- Integração com outros módulos

### FASE 3: Integração
1. Atualizar `__init__.py` do pacote
2. Adicionar ao índice de documentação
3. Verificar imports funcionam

---

## FORMATO DE SAÍDA

```markdown
# 🆕 Novo Módulo: [nome_modulo]

## Arquivos Criados

### 📄 [caminho/arquivo.py]
```python
[código completo]
```

### 📄 [caminho/test_arquivo.py]
```python
[testes completos]
```

### 📄 [caminho/docs.md]
```markdown
[documentação completa]
```

## Atualizações em Arquivos Existentes

### 📝 [arquivo_modificado]
```diff
+ linha adicionada
```
```

---

## RESTRIÇÕES

- ✅ SIGA os padrões existentes do projeto
- ✅ INCLUA docstrings e type hints
- ✅ CRIE testes funcionais (não placeholders)
- ✅ EXECUTE imediatamente ao receber este prompt
