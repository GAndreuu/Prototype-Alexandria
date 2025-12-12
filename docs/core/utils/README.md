# 📦 Utils - Documentação

**Utilitários do sistema Alexandria**

---

## 🌾 Harvester

**Module**: `core/utils/harvester.py` (126 LOC)

Colheitadeira automática de papers científicos via Arxiv.

```python
from core.utils.harvester import ArxivHarvester

harvester = ArxivHarvester()

# Buscar papers
papers = harvester.search_papers("neural compression", max_results=10)

# Harvest completo: busca → download → ingestão
harvester.harvest(
    queries=["VQ-VAE", "predictive coding"],
    max_per_query=5,
    ingest=True  # Ingere no LanceDB
)
```

---

## 🤖 Local LLM

**Module**: `core/utils/local_llm.py` (203 LOC)

Expert Tático com TinyLlama-1.1B para RAG local (zero API cost).

```python
from core.utils.local_llm import LocalLLM

llm = LocalLLM()  # TinyLlama-1.1B-Chat

# Sintetizar resposta a partir de evidências
response = llm.synthesize_facts(
    query="Como funciona VQ-VAE?",
    evidence=retrieved_chunks,
    max_length=512
)
```

**Otimizações**:
- Float32 em CPU (mais rápido que FP16 emulado)
- Multi-threading (8 threads)
- Fallback para concatenação se modelo falhar

---

## 📝 Logger

**Module**: `core/utils/logger.py` (45 LOC)

Logger estruturado com Loguru.

```python
from core.utils.logger import logger

logger.info("Mensagem informativa")
logger.debug("Debug detalhado")  # Só em arquivo
logger.error("Erro crítico")
```

**Saídas**:
- Console: colorido, nível INFO+
- `data/logs/system.log`: JSON estruturado, DEBUG+
- `data/logs/system_readable.log`: texto legível, INFO+

---

**Last Updated**: 2025-12-07
