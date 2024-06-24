import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model:  int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, ::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, embedding):
        embedding = embedding + (self.pe[:, :embedding.shape[1], :]).requires_grad_(False)
        return self.dropout(embedding)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        means = torch.mean(x, dim=-1, keepdim=True)  # TODO: Check the shapes of the inputs to this layer. [Batch, Seq, d_model]
        variance = torch.var(x, dim=-1, keepdim=True)
        return self.alpha*(x - means)/torch.sqrt(variance + self.eps) + self.bias


class CausalAttention(nn.Module):
  def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool =False):
    super().__init__()
    self.d_out = d_out 
    self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.triu(torch.ones((context_length, context_length), dtype=torch.bool), diagonal=1))

  def forward(self, inputs):
    K = self.W_k(inputs)
    Q = self.W_q(inputs)
    V = self.W_v(inputs)
    attention_scores = Q @ K.transpose(1,2)
    attention_scores.masked_fill_(self.mask, -torch.inf)
    attention_weights = self.dropout(torch.softmax(attention_scores/self.d_out**0.5, dim=-1))
    context_matrix = attention_weights @ V 
    return context_matrix


class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

  def forward(self, inputs):
    return torch.cat([head(inputs) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads 
    self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.triu(torch.ones((context_length, context_length), dtype=torch.bool), diagonal=1))

  def forward(self, inputs):
    batches, num_tokens, d_in = inputs.shape
    K = self.W_k(inputs)
    Q = self.W_q(inputs)
    V = self.W_v(inputs)

    K = K.view(batches, num_tokens, self.num_heads, self.head_dim)
    Q = Q.view(batches, num_tokens, self.num_heads, self.head_dim)
    V = V.view(batches, num_tokens, self.num_heads, self.head_dim)

    K = K.transpose(1, 2)
    Q = Q.transpose(1, 2)
    V = V.transpose(1, 2)

    attention_scores = Q @ K.transpose(2, 3)
    attention_scores.masked_fill_(self.mask, -torch.inf)
    attention_weights = self.dropout(torch.softmax(attention_scores/K.shape[-1]**0.5, dim=-1))
    context_matrix = (attention_weights @ V).transpose(1, 2)
    context_matrix = context_matrix.contiguous().view(batches, num_tokens, self.d_out)

    return self.out_proj(context_matrix)


class DummyGPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = DummyLayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape 
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds 
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits


class DummyTransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()

  def forward(self, x):
    return x 


class DummyLayerNorm(nn.Module):
  def __init__(self, normalized_shape, eps=1e-5):
    super().__init__()

  def forward(self, x):
    return x
