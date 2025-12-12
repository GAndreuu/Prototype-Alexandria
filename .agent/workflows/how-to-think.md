---
description: Define o protocolo cognitivo e a Cadeia de Pensamento (Chain of Thought) para resolução de tarefas.
---

# Workflow: How to Think (Protocolo Cognitivo)

Este documento estabelece o padrão mental obrigatório para o agente ao abordar qualquer tarefa no sistema Alexandria.

## O Processo: Chain of Thought

Para garantir segurança e eficácia, você deve seguir estritamente este fluxo de 3 etapas para cada solicitação:

### 1. Identificar e Compreender (Understand)
Antes de escrever qualquer linha de código ou executar comandos:
- Qual é o objetivo final do usuário?
- Quais módulos do sistema serão afetados?
- Existe alguma restrição explícita ou implícita?

> **Regra**: Se houver ambiguidade, PARE e peça clarificação.

### 2. Pesquisar e Mapear (Search & Map)
Você não deve "alucinar" a estrutura do projeto. Você deve vê-la.

**CRÍTICO**: Antes de fazer alterações, você DEVE sempre usar a ferramenta `search_codebase` (ou ferramentas de busca equivalentes como `grep_search` e `find_by_name`) para localizar o código relevante.

- Não adivinhe nomes de arquivos.
- Não adivinhe assinaturas de funções.
- Não adivinhe onde as classes estão definidas.
- **Busque, Leia, Confirme.**

### 3. Decompor e Planejar (Breakdown)
Quebre a solução em passos atômicos e lógicos.
- Liste os arquivos que precisam ser criados ou modificados.
- Defina a ordem de execução.
- Antecipe possíveis efeitos colaterais.

### 4. Executar (Execute)
Apenas após validar o plano mentalmente:
- Aplique as mudanças.
- Use os workflows existentes (ex: `/criar-feature`) se aplicável.
- Valide o resultado.

---

> "Efficiency is doing things right; Effectiveness is doing the right things."
