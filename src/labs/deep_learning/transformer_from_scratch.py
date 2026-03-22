"""
Lab: Deep Learning - Transformer from Scratch
===============================================
A complete, minimal Transformer Encoder implementation in PyTorch,
showing every internal component: positional encoding, multi-head
attention, feed-forward network, and layer normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Core attention mechanism: softmax(QK^T / sqrt(d_k)) V"""

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: project Q,K,V into h subspaces, attend, concat."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, weights = self.attention(Q, K, V, mask)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.dropout(self.W_o(output)), weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network (2-layer MLP with GELU)."""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer: MHA + Add&Norm + FFN + Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual connection
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder with N stacked layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(self, src, mask=None):
        # Embed + scale + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


# =============================================================================
# DEMO: Text Classification with our Transformer
# =============================================================================
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 10000
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    SEQ_LEN = 32
    BATCH_SIZE = 4
    NUM_CLASSES = 5

    # Create model
    encoder = TransformerEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_MODEL * 4,
    )
    classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    # Dummy input
    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    encoded, attn_weights = encoder(src)
    # Use [CLS]-like aggregation (mean pooling)
    pooled = encoded.mean(dim=1)
    logits = classifier(pooled)

    print(f"Input shape:              {src.shape}")
    print(f"Encoded shape:            {encoded.shape}")
    print(f"Pooled shape:             {pooled.shape}")
    print(f"Logits shape:             {logits.shape}")
    print(f"Attention weights shapes: {[w.shape for w in attn_weights]}")
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")
