import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, **kwargs):
    super().__init__(**kwargs)
    assert d_out % num_heads == 0
    self.W_q = nn.Linear(d_in, d_out, qkv_bias)
    self.W_k = nn.Linear(d_in, d_out, qkv_bias)
    self.W_v = nn.Linear(d_in, d_out, qkv_bias)

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads

    self.W_o = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)

  def forward(self, inputs):
    b, num_tokens, d_in = inputs.shape
    Q, K, V = self.W_q(inputs), self.W_k(inputs), self.W_v(inputs)
    Q = Q.reshape((Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)).transpose(-2, -3)
    K = K.reshape((K.shape[0], K.shape[1], self.num_heads, self.head_dim)).transpose(-2, -3)
    V = V.reshape((V.shape[0], V.shape[1], self.num_heads, self.head_dim)).transpose(-2, -3)

    attn_scores = Q @ K.transpose(-1, -2) / self.head_dim**0.5
    attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -1e6)
    attention_weights = torch.softmax(attn_scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    attention = attention_weights @ V
    attention = attention.contiguous().reshape((inputs.shape[0], inputs.shape[1], self.d_out))
    return self.W_o(attention)


class LayerNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-5):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
    self.eps = eps

  def forward(self, x):
    return self.scale * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) +self.shift


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
    self.layernorm1 = LayerNorm(cfg["emb_dim"])
    self.MultiHeadAttention =  MultiHeadAttention(cfg["emb_dim"], 
                                                  cfg["emb_dim"], 
                                                  cfg["context_length"], 
                                                  cfg["drop_rate"], 
                                                  cfg["n_heads"], 
                                                  cfg["qkv_bias"])
    self.dropout1 = nn.Dropout(cfg["drop_rate"])

    self.layernorm2 = LayerNorm(cfg["emb_dim"])
    self.FeedForwardBlock = FeedForward(cfg)
    self.dropout2 = nn.Dropout(cfg["drop_rate"])
    
  def forward(self, x):
    x = self.dropout1(self.MultiHeadAttention(self.layernorm1(x))) + x
    x = self.dropout2(self.FeedForwardBlock(self.layernorm2(x))) + x
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


