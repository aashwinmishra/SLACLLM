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


class SelfAttentionV1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.d_out = d_out
    self.W_k = nn.Parameter(torch.randn(d_in,d_out))
    self.W_q = nn.Parameter(torch.randn(d_in,d_out))
    self.W_v = nn.Parameter(torch.randn(d_in,d_out))

  def forward(self, inputs):
    K = inputs @ self.W_k
    Q = inputs @ self.W_q
    V = inputs @ self.W_v

    attention_weights = torch.softmax(Q @ K.T/self.d_out**0.5, dim=-1)
    out = attention_weights @ V
    return attention_weights, out


class SelfAttentionV2(nn.Module):
  def __init__(self, d_in: int, d_out: int, kqv_bias: bool=False):
    super().__init__()
    self.W_k = nn.Linear(d_in, d_out, bias=kqv_bias)
    self.W_q = nn.Linear(d_in, d_out, bias=kqv_bias)
    self.W_v = nn.Linear(d_in, d_out, bias=kqv_bias)
    self.d_out = d_out 

  def forward(self, inputs: torch.Tensor):
    K = self.W_k(inputs)
    Q = self.W_q(inputs)
    V = self.W_v(inputs)

    attention_weights = torch.softmax(Q @ K.T/self.d_out**0.5, dim=-1)
    context_matrix = attention_weights @ V 
    return attention_weights, context_matrix
