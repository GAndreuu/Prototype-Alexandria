# Next Step: Monolith 512D Integration

## O Que Aconteceu

Tentamos integrar um novo encoder VQ-VAE (512D hidden dimension) para melhorar a utilização do codebook de 16% para 100%.

### Experimento Realizado

1. **Treinamos novo modelo:**
   - 128,138 embeddings
   - 20 epochs
   - Final loss: 0.00234
   - Arquitetura: 384D → 512D → 4×256 codes

2. **Integração:**
   - Criado `core/encoders/monolith_encoder.py`
   - Atualizado `mycelial_reasoning.py`
   - Resetado estado Micelial

### Problema Descoberto

**Colapso total do codebook:**
- Head 0: Apenas código 163 (5000/5000 = 100%)
- Head 1: Apenas código 158
- Head 2: Apenas código 145  
- Head 3: Apenas 1 código

**Utilização real: 0.4% (4/1024 códigos)**

O modelo **não aprendeu** a usar os codebooks. Todos os embeddings são mapeados para os mesmos 4 códigos.

### Root Cause

Possíveis causas:
1. EMA update dos codebooks não funcionou
2. Learning rate muito baixo
3. Commitment loss muito fraco
4. Convergiu para mínimo local trivial

### Arquivos Salvos em `next_passo/`

- `monolith_export/` — Pesos treinados (8.4 MB)
- `encoders/` — Código do encoder inference
- `train_monolith.py` — Script de treinamento
- `test_monolith_integration.py` — Testes
- `populate_mycelial.py` — Populador de rede
- `system_health_check.py` — Diagnóstico

### Próximos Passos (Para Resolver)

#### Opção 1: Fix de Training
- Aumentar commitment loss weight (0.25 → 1.0)
- Adicionar codebook reset para códigos mortos
- Usar learning rate schedule
- Adicionar "perplexity" loss para forçar diversidade

#### Opção 2: Arquitetura Alternativa
- Tentar Gumbel-Softmax quantizer
- Usar VQ-VAE-2 (hierarchical)
- Implementar K-means initialization

#### Opção 3: Pesos Pré-treinados
- Procurar modelo VQ-VAE pré-treinado
- Fine-tune em vez de treinar do zero

### Sistema Atual (Restaurado)

Voltamos para:
- **MonolithV13** (256D hidden)
- **16% codebook usage** (não ideal, mas funciona)
- Estado Micelial restaurado

### Referências Úteis

1. [VQ-VAE Paper](https://arxiv.org/abs/1711.00937)
2. [Fixing codebook collapse](https://github.com/rosinality/vq-vae-2-pytorch)
3. EMA codebook updates: α decay schedule importante

### Código de Treinamento Modificado (Ideia)

```python
# No ProductQuantizer, adicionar:

# Reset códigos mortos a cada N steps
if step % reset_interval == 0:
    usage = (ema_count > threshold)
    dead = ~usage
    if dead.any():
        # Reinicializar códigos mortos com embeddings reais
        dead_idx = dead.nonzero()
        self.codebooks[dead_idx] = random_sample_from_data()
        
# Adicionar perplexity loss
perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10)))
loss += 0.1 * (target_perplexity - perplexity) ** 2
```

### Métricas de Sucesso (Para Próxima Vez)

- [ ] Codebook usage > 80% por head
- [ ] Power-law α entre 1.2-2.0
- [ ] R² do fit > 0.9
- [ ] Reconstruction loss < 0.01
- [ ] Nenhum código com >10% do total

---

**Data:** 2025-12-01  
**Status:** FAILED - Codebook collapse  
**Rollback:** ✅ Sistema restaurado para versão funcionando
