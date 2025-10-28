# GPT Model 详细分析

## 一、模型配置 (GPT-2 124M)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数量
    "n_layers": 12,         # Transformer层数量
    "drop_rate": 0.1,       # Dropout率
    "qkv_bias": False       # Query-Key-Value偏置
}
```

## 二、完整的 GPTModel 类代码

```python
import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Token嵌入层: 将token ID映射到embedding向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置嵌入层: 为每个位置添加位置信息
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Embedding dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 12个Transformer Block堆叠
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 最终层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出头: 将embedding映射回词汇表大小
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

## 三、TransformerBlock 类代码

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头注意力机制
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

        # 前馈神经网络
        self.ff = FeedForward(cfg)

        # 两个层归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Shortcut connection的dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接

        return x
```

## 四、FeedForward 类代码

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展到4倍
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压缩回原始维度
        )

    def forward(self, x):
        return self.layers(x)
```

## 五、LayerNorm 类代码

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

## 六、MultiHeadAttention 类代码

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # Query, Key, Value投影
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 输出投影
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 因果掩码（Causal Mask）
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑为多头格式
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
```

## 七、GPT Model 架构与数据流图

```
输入: Token IDs [batch_size, seq_len]
    |
    ├─────────────────────────────────────┐
    |                                     |
    v                                     v
Token Embedding                   Position Embedding
(50257 → 768)                     (1024 → 768)
    |                                     |
    └──────────────┬──────────────────────┘
                   | 相加
                   v
              Dropout (0.1)
                   |
                   v
    ╔══════════════════════════════════╗
    ║   Transformer Block 1            ║
    ╠══════════════════════════════════╣
    ║                                  ║
    ║  ┌───────────────────┐           ║
    ║  │  LayerNorm 1      │           ║
    ║  └─────────┬─────────┘           ║
    ║            │                     ║
    ║            v                     ║
    ║  ┌─────────────────────┐         ║
    ║  │ Multi-Head Attention│         ║
    ║  │   (12 heads)        │         ║
    ║  │                     │         ║
    ║  │  Q, K, V Projection │         ║
    ║  │  (768 → 768)        │         ║
    ║  │  Head dim: 768/12=64│         ║
    ║  │                     │         ║
    ║  │  Output Projection  │         ║
    ║  │  (768 → 768)        │         ║
    ║  └─────────┬───────────┘         ║
    ║            │                     ║
    ║        Dropout (0.1)             ║
    ║            │                     ║
    ║  ┌─────────┴────────┐            ║
    ║  │ Residual Add (+) │◄───────────╢ Shortcut
    ║  └─────────┬────────┘            ║
    ║            │                     ║
    ║  ┌─────────v─────────┐           ║
    ║  │  LayerNorm 2      │           ║
    ║  └─────────┬─────────┘           ║
    ║            │                     ║
    ║            v                     ║
    ║  ┌─────────────────────┐         ║
    ║  │  Feed Forward Net   │         ║
    ║  │                     │         ║
    ║  │  Linear (768→3072)  │         ║
    ║  │  GELU Activation    │         ║
    ║  │  Linear (3072→768)  │         ║
    ║  └─────────┬───────────┘         ║
    ║            │                     ║
    ║        Dropout (0.1)             ║
    ║            │                     ║
    ║  ┌─────────┴────────┐            ║
    ║  │ Residual Add (+) │◄───────────╢ Shortcut
    ║  └─────────┬────────┘            ║
    ║            │                     ║
    ╚════════════╪════════════════════╝
                 │
                 v
              重复 12 次
                 │
                 v
    ╔══════════════════════════════════╗
    ║   Transformer Block 12           ║
    ╚════════════╪════════════════════╝
                 │
                 v
           LayerNorm (Final)
                 |
                 v
           Output Head
           (768 → 50257)
                 |
                 v
       Logits [batch_size, seq_len, 50257]
```

## 八、参数数量详细计算

### 8.1 Token Embedding 层
- 参数: `vocab_size × emb_dim = 50257 × 768 = 38,597,376`

### 8.2 Position Embedding 层
- 参数: `context_length × emb_dim = 1024 × 768 = 786,432`

### 8.3 单个 Transformer Block 参数

#### 多头注意力 (MultiHeadAttention)
- W_query: `768 × 768 = 589,824`
- W_key: `768 × 768 = 589,824`
- W_value: `768 × 768 = 589,824`
- out_proj: `768 × 768 + 768 = 590,592` (包含bias)
- **小计: 2,360,064**

#### LayerNorm 1
- scale: `768`
- shift: `768`
- **小计: 1,536**

#### FeedForward Network
- Linear1: `768 × 3072 + 3072 = 2,362,368`
- Linear2: `3072 × 768 + 768 = 2,360,064`
- **小计: 4,722,432**

#### LayerNorm 2
- scale: `768`
- shift: `768`
- **小计: 1,536**

#### 单个Block总计
- **7,085,568 参数**

### 8.4 12个 Transformer Blocks
- `12 × 7,085,568 = 85,026,816`

### 8.5 Final LayerNorm
- scale: `768`
- shift: `768`
- **小计: 1,536**

### 8.6 Output Head
- 参数: `768 × 50257 = 38,597,376`

### 8.7 总参数数量

**不使用 Weight Tying:**
```
Token Embedding:        38,597,376
Position Embedding:        786,432
Transformer Blocks:     85,026,816
Final LayerNorm:             1,536
Output Head:            38,597,376
─────────────────────────────────
总计:                  163,009,536 参数
```

**使用 Weight Tying (Token Embedding = Output Head):**
```
Token Embedding:        38,597,376 (共享)
Position Embedding:        786,432
Transformer Blocks:     85,026,816
Final LayerNorm:             1,536
Output Head:                     0 (共享Token Embedding)
─────────────────────────────────
总计:                  124,412,160 参数
```

## 九、模型内存占用

### 使用 float32 (4 bytes per parameter):
- **不使用 Weight Tying:** `163,009,536 × 4 = 652,038,144 bytes ≈ 621.83 MB`
- **使用 Weight Tying:** `124,412,160 × 4 = 497,648,640 bytes ≈ 474.70 MB`

### 使用 float16 (2 bytes per parameter):
- **不使用 Weight Tying:** `163,009,536 × 2 = 326,019,072 bytes ≈ 310.91 MB`
- **使用 Weight Tying:** `124,412,160 × 2 = 248,824,320 bytes ≈ 237.35 MB`

## 十、数据流维度变化

### 输入阶段
```
Input Token IDs:        [batch_size, seq_len]
Token Embeddings:       [batch_size, seq_len, 768]
Position Embeddings:    [batch_size, seq_len, 768]
Combined:               [batch_size, seq_len, 768]
```

### Transformer Block 内部
```
输入:                   [batch_size, seq_len, 768]
LayerNorm1:             [batch_size, seq_len, 768]

Multi-Head Attention:
  Q, K, V 投影:         [batch_size, seq_len, 768]
  分割为12个头:          [batch_size, 12, seq_len, 64]
  注意力输出:            [batch_size, 12, seq_len, 64]
  合并头:               [batch_size, seq_len, 768]
  输出投影:              [batch_size, seq_len, 768]

残差连接后:              [batch_size, seq_len, 768]
LayerNorm2:             [batch_size, seq_len, 768]

Feed Forward:
  Linear1:              [batch_size, seq_len, 3072]
  GELU:                 [batch_size, seq_len, 3072]
  Linear2:              [batch_size, seq_len, 768]

残差连接后:              [batch_size, seq_len, 768]
```

### 输出阶段
```
Final LayerNorm:        [batch_size, seq_len, 768]
Output Head:            [batch_size, seq_len, 50257]
```

## 十一、关键设计特点

1. **残差连接 (Residual Connections)**: 在注意力和前馈网络前后都有残差连接，帮助梯度流动
2. **Pre-Norm架构**: LayerNorm放在子层之前（现代Transformer的标准做法）
3. **因果掩码 (Causal Mask)**: 确保模型只能看到当前位置之前的token
4. **多头注意力**: 12个注意力头，每个头维度为64 (768/12)
5. **前馈网络扩展**: 中间层维度扩展到4倍 (768 → 3072 → 768)
6. **GELU激活**: 使用GELU而不是ReLU，提供更平滑的激活
7. **Dropout正则化**: 在多个位置使用10%的dropout防止过拟合

## 十二、与原始GPT-2的差异

1. **Weight Tying**: 原始GPT-2使用weight tying（共享token embedding和output layer），得到124M参数
2. **本实现**: 为了训练便利性，没有使用weight tying，得到163M参数
3. **兼容性**: 后续章节会展示如何加载预训练的GPT-2权重

## 十三、总结

这个GPT-2 124M模型的核心特点：
- **实际参数**: 163,009,536 (不使用weight tying) 或 124,412,160 (使用weight tying)
- **主要组件**: Token/Position Embedding + 12个Transformer Blocks + Output Head
- **每个Block**: Multi-Head Attention + Feed Forward Network + LayerNorm × 2 + Residual Connections
- **设计理念**: 使用残差连接和层归一化确保深层网络的稳定训练
