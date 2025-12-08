# 📝 PROMPT: Criar Documentação Completa

---

## INSTRUÇÃO PARA O AGENTE

Crie documentação técnica completa para este projeto do zero, seguindo o protocolo abaixo.

---

## PROTOCOLO DE EXECUÇÃO

### FASE 1: Análise Profunda
1. Varra toda a estrutura de arquivos
2. Leia os arquivos principais de código
3. Identifique funções, classes e módulos públicos
4. Mapeie o fluxo de dados e dependências

### FASE 2: Geração de Documentos

Crie os seguintes arquivos:

#### 📄 README.md (Raiz)
```markdown
# [Nome do Projeto]

## Visão Geral
[Descrição clara do propósito]

## Início Rápido
[Instalação + primeiro uso em 3 passos]

## Estrutura do Projeto
[Árvore de diretórios com descrições]

## Módulos
[Lista com links para docs detalhadas]

## Tecnologias
[Stack utilizada]

## Contribuição
[Como contribuir]
```

#### 📄 docs/ARCHITECTURE.md
```markdown
# Arquitetura Técnica

## Diagrama de Componentes
[Mermaid ou ASCII]

## Fluxo de Dados
[Como os dados fluem pelo sistema]

## Decisões de Design
[ADRs - Architecture Decision Records]
```

#### 📄 docs/API.md (se aplicável)
```markdown
# Referência da API

## Endpoints / Funções Públicas
[Lista completa com parâmetros e retornos]

## Exemplos de Uso
[Código funcional]
```

#### 📄 docs/SETUP.md
```markdown
# Guia de Configuração

## Pré-requisitos
[Dependências necessárias]

## Instalação Passo a Passo
[Comandos exatos]

## Variáveis de Ambiente
[Lista completa de configs]

## Troubleshooting
[Problemas comuns e soluções]
```

### FASE 3: Indexação

#### 📄 docs/INDEX.md
```markdown
# Índice da Documentação

1. [README](../README.md) - Visão geral
2. [Arquitetura](./ARCHITECTURE.md) - Design técnico
3. [API](./API.md) - Referência de código
4. [Setup](./SETUP.md) - Configuração
```

---

## FORMATO DE SAÍDA

Para cada arquivo gerado, apresente:

```
📄 [CAMINHO_DO_ARQUIVO]

[CONTEÚDO COMPLETO DO ARQUIVO]

---
```

---

## RESTRIÇÕES

- ❌ NÃO use placeholders genéricos como "[inserir aqui]"
- ❌ NÃO invente funcionalidades que não existem no código
- ✅ BASEIE tudo no código real analisado
- ✅ INCLUA exemplos de código reais do projeto
- ✅ CRIE arquivos prontos para uso (copy-paste direto)
- ✅ EXECUTE imediatamente ao receber este prompt
