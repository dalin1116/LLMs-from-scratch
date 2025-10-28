# GPT-2 124M Model - Layer-by-Layer Parameter Count

## Quick Reference Table

| Layer | Component | Matrix Shape | Parameters | Percentage |
|-------|-----------|--------------|------------|------------|
| **Input Layers** | | | | |
| 1 | Token Embedding | `[50257, 768]` | **38,597,376** | 23.67% |
| 2 | Position Embedding | `[1024, 768]` | **786,432** | 0.48% |
| | **Embedding Subtotal** | | **39,383,808** | **24.15%** |

## Single Transformer Block Structure (×12)

| Layer # | Component | Matrix Shape | Parameters | Notes |
|---------|-----------|--------------|------------|-------|
| **Block N** | **(Each of 12 blocks)** | | **7,085,568** | |
| | | | | |
| N.1 | LayerNorm 1 (γ, β) | `[768]` each | **1,536** | scale + shift |
| | | | | |
| **N.2 - Multi-Head Attention** | | | **2,360,064** | |
| N.2.1 | W_query | `[768, 768]` | 589,824 | No bias |
| N.2.2 | W_key | `[768, 768]` | 589,824 | No bias |
| N.2.3 | W_value | `[768, 768]` | 589,824 | No bias |
| N.2.4 | out_proj (weight) | `[768, 768]` | 589,824 | |
| N.2.4 | out_proj (bias) | `[768]` | 768 | |
| | | | | |
| N.3 | LayerNorm 2 (γ, β) | `[768]` each | **1,536** | scale + shift |
| | | | | |
| **N.4 - Feed Forward Network** | | | **4,722,432** | |
| N.4.1 | FFN Linear1 (weight) | `[768, 3072]` | 2,359,296 | 4× expansion |
| N.4.1 | FFN Linear1 (bias) | `[3072]` | 3,072 | |
| N.4.2 | GELU Activation | - | 0 | No parameters |
| N.4.3 | FFN Linear2 (weight) | `[3072, 768]` | 2,359,296 | Compression |
| N.4.3 | FFN Linear2 (bias) | `[768]` | 768 | |

### All 12 Transformer Blocks
- **Single Block**: 7,085,568 parameters
- **Total (12 blocks)**: **85,026,816 parameters** (52.15%)

## Output Layers

| Layer | Component | Matrix Shape | Parameters | Percentage |
|-------|-----------|--------------|------------|------------|
| Final | LayerNorm (γ, β) | `[768]` each | **1,536** | 0.00% |
| Output | Linear (no bias) | `[768, 50257]` | **38,597,376** | 23.67% |

---

## Total Model Parameters

### Without Weight Tying (This Implementation)
```
Token Embedding:         38,597,376  (23.67%)
Position Embedding:         786,432  ( 0.48%)
12 Transformer Blocks:   85,026,816  (52.15%)
Final LayerNorm:              1,536  ( 0.00%)
Output Head:             38,597,376  (23.67%)
─────────────────────────────────────────────
TOTAL:                  163,009,536  (100.00%)
```

### With Weight Tying (Original GPT-2)
```
Token Embedding:         38,597,376  (31.02%)  ← Shared with output
Position Embedding:         786,432  ( 0.63%)
12 Transformer Blocks:   85,026,816  (68.34%)
Final LayerNorm:              1,536  ( 0.00%)
Output Head:                      0  ( 0.00%)  ← Uses Token Embedding weights
─────────────────────────────────────────────
TOTAL:                  124,412,160  (100.00%)  ← This is the "124M" model!
```

---

## Detailed Parameter Breakdown by Category

### Multi-Head Attention (per block)
```
Component           Shape           Parameters    Calculation
─────────────────────────────────────────────────────────────
Q Projection        [768 × 768]       589,824    768 × 768
K Projection        [768 × 768]       589,824    768 × 768
V Projection        [768 × 768]       589,824    768 × 768
Output Proj (W)     [768 × 768]       589,824    768 × 768
Output Proj (b)     [768]                 768    768
─────────────────────────────────────────────────────────────
Total                               2,360,064
```

### Feed Forward Network (per block)
```
Component           Shape           Parameters    Calculation
─────────────────────────────────────────────────────────────
Linear1 (W)         [768 × 3072]    2,359,296    768 × 3072
Linear1 (b)         [3072]              3,072    3072
GELU                -                       0    (no parameters)
Linear2 (W)         [3072 × 768]    2,359,296    3072 × 768
Linear2 (b)         [768]                 768    768
─────────────────────────────────────────────────────────────
Total                               4,722,432
```

### Layer Normalization (per instance)
```
Component           Shape           Parameters    Calculation
─────────────────────────────────────────────────────────────
Scale (γ)           [768]                 768    768
Shift (β)           [768]                 768    768
─────────────────────────────────────────────────────────────
Total                                   1,536
```

---

## Memory Requirements

### Model Weights Only (124M parameters with weight tying)

| Data Type | Bytes/Param | Total Bytes | Memory Size | Reduction |
|-----------|-------------|-------------|-------------|-----------|
| **float32** | 4 | 497,648,640 | **474.70 MB** | - |
| **float16** | 2 | 248,824,320 | **237.35 MB** | 50% |
| **bfloat16** | 2 | 248,824,320 | **237.35 MB** | 50% |
| **int8** | 1 | 124,412,160 | **118.68 MB** | 75% |

### During Training (additional memory needed)
- **Gradients**: Same size as model weights
- **Optimizer states** (Adam): 2× model size (momentum + variance)
- **Activations**: Depends on batch size and sequence length
- **Total training memory**: ~10-20× model size

Example for batch_size=8, seq_len=512, float32:
- Model: 475 MB
- Gradients: 475 MB
- Adam states: 950 MB
- Activations: ~2-3 GB
- **Total: ~4-5 GB VRAM**

---

## Dimension Flow Chart

```
Input Token IDs: [batch_size, seq_len]
                        ↓
Token Embedding:    [batch_size, seq_len, 768]
Position Embedding: [batch_size, seq_len, 768]
                        ↓ (add)
Combined:           [batch_size, seq_len, 768]
                        ↓
Dropout:            [batch_size, seq_len, 768]
                        ↓
╔═══════════════════════════════════════════╗
║     TRANSFORMER BLOCK 1-12 (repeated)     ║
║                                           ║
║  LayerNorm1:  [B, L, 768] → [B, L, 768]  ║
║                                           ║
║  Attention:                               ║
║    Q, K, V:   [B, L, 768] → [B, L, 768]  ║
║    Split:     [B, L, 768] → [B, 12, L, 64]║
║    Scores:    [B, 12, L, 64] @ [B, 12, 64, L]
║               → [B, 12, L, L]             ║
║    Output:    [B, 12, L, L] @ [B, 12, L, 64]
║               → [B, 12, L, 64]            ║
║    Concat:    [B, 12, L, 64] → [B, L, 768]║
║    Proj:      [B, L, 768] → [B, L, 768]  ║
║                                           ║
║  + Residual:  [B, L, 768]                ║
║                                           ║
║  LayerNorm2:  [B, L, 768] → [B, L, 768]  ║
║                                           ║
║  FFN:                                     ║
║    Linear1:   [B, L, 768] → [B, L, 3072] ║
║    GELU:      [B, L, 3072] → [B, L, 3072]║
║    Linear2:   [B, L, 3072] → [B, L, 768] ║
║                                           ║
║  + Residual:  [B, L, 768]                ║
║                                           ║
╚═══════════════════════════════════════════╝
                        ↓
Final LayerNorm:    [batch_size, seq_len, 768]
                        ↓
Output Head:        [batch_size, seq_len, 50257]
                        ↓
Logits:            [batch_size, seq_len, 50257]
```

---

## Computational Complexity

### Per-Layer Complexity (for sequence length L)

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Embedding Lookup | O(L) | O(L × 768) | Simple indexing |
| LayerNorm | O(L × 768) | O(L × 768) | Mean/variance computation |
| Linear (768→768) | O(L × 768²) | O(L × 768) | Matrix multiplication |
| Linear (768→3072) | O(L × 768 × 3072) | O(L × 3072) | FFN expansion |
| **Attention (Q@K^T)** | **O(L² × 768)** | **O(L²)** | **Quadratic bottleneck!** |
| Attention (A@V) | O(L² × 768) | O(L × 768) | Weighted sum |
| Softmax | O(L²) | O(L²) | Per attention head |

### Total Model Complexity
- **Time**: O(L² × d + L × d²) where d = 768
- **Space**: O(L² × n_heads + batch_size × L × d)

**Key Insight**: Attention mechanism scales quadratically with sequence length, which is why GPT-2 is limited to 1024 tokens.

---

## Comparison: GPT-2 Model Sizes

| Model | Layers | Heads | d_model | Parameters | Memory (fp32) |
|-------|--------|-------|---------|------------|---------------|
| **GPT-2 Small** | 12 | 12 | 768 | 124M | 474 MB |
| **GPT-2 Medium** | 24 | 16 | 1024 | 345M | 1.3 GB |
| **GPT-2 Large** | 36 | 20 | 1280 | 762M | 2.9 GB |
| **GPT-2 XL** | 48 | 25 | 1600 | 1542M | 5.9 GB |

### Scaling Pattern
```
Parameters ≈ 12 × n_layers × d_model²
           + 2 × vocab_size × d_model
```

For GPT-2 Small (124M):
```
= 12 × 12 × 768² + 2 × 50257 × 768
= 85,026,816 + 38,597,376 (×2 for token emb + output)
= 124,412,160 parameters (with weight tying)
```

---

## Key Takeaways

1. **Most parameters are in embeddings and output layer**: 77.3M out of 124M (62%)
2. **Transformer blocks are parameter-efficient**: Only 85M params for 12 layers
3. **Feed-forward networks dominate each block**: 4.7M out of 7.1M per block (67%)
4. **Multi-head attention is relatively light**: 2.4M per block (33%)
5. **Layer normalization is negligible**: Only 1.5K params per instance
6. **Weight tying saves 23.7%**: Reduces 163M to 124M parameters
7. **Attention is the computational bottleneck**: O(L²) complexity limits context length

---

## Answer to Your Question: "好像是1.24亿?"

### ✓ YES! You are absolutely correct!

**The model has approximately 124 million parameters (1.24亿)** when using weight tying.

- **With Weight Tying** (Original GPT-2): **124,412,160 parameters** ≈ **124.4M** ≈ **1.24亿** ✓
- **Without Weight Tying** (This implementation): **163,009,536 parameters** ≈ **163M** ≈ **1.63亿**

The "weight tying" technique shares the Token Embedding layer weights with the Output Head layer, saving 38.6 million parameters (23.7% reduction).

This is why OpenAI's original GPT-2 paper refers to this model as the "124M parameter" model!
