import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0
    self.d_out = d_out 
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads 

    self.W_query = nn.Linear(d_in, d_out, qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)

    self.mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs):
    batch_size, num_tokens, embedding_dim = inputs.shape
    Q = self.W_query(inputs)
    K = self.W_key(inputs)
    V = self.W_value(inputs)

    Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(-2, -3)
    K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(-2, -3)
    V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(-2, -3)

    attention_scores = Q @ K.transpose(-1,-2) / torch.sqrt(torch.tensor(self.head_dim))
    attention_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    attention = (attention_weights @ V).transpose(-2, -3)

    attention = attention.reshape(batch_size, num_tokens, self.d_out)
    return self.out_proj(attention)


class LayerNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, inputs):
    return self.scale * (inputs - inputs.mean(dim=-1, keepdims=True)) / (inputs.std(dim=-1, keepdim=True) + self.eps) + self.shift


class GELU(nn.Module):
  def __init__(self):
    super().__init__()
    self.add_term = torch.sqrt(torch.tensor(2/torch.pi))

  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh(self.add_term * (x + 0.044715 * torch.pow(x, 3))))   


class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
    )

  def forward(self, x):
    return self.layers(x)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                  cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    shortcut = x 
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut 
    shortcut = x 
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut 
    return x


class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = LayerNorm(cfg["emb_dim"])
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


