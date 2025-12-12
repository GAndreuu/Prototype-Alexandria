---
description: Atualiza o mapa de contexto topológico (Cosmic Garden) para permitir navegação inteligente.
---

# Workflow: Mapear Contexto

Este workflow executa o `context_mapper.py` para gerar um índice invertido de todo o projeto. Isso serve como base para funcionalidades de busca inteligente e "gravidade de contexto".

## Passos

1. Execute o script de mapeamento:
```bash
python3 .agent/scripts/context_mapper.py
```

2. Verifique se o arquivo de mapa foi gerado com sucesso em `.agent/context_map.json`.

3. (Opcional) Leia o resumo do JSON gerado para confirmar quantos arquivos foram indexados.

---

> "O mapa não é o território, mas é a melhor ferramenta para não se perder nele."
