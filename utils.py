import torch
import torch.nn as nn
import tiktoken


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def text_to_token_ids(text, tokenizer):
  ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  return torch.tensor(ids).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
  return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)
    logits = logits[:,-1, :]
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probs, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=-1)
  return idx


def temperature_scaled_sampling(model, idx, max_new_tokens, context_size, temperature: float=1.0):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)
    logits = logits[:,-1, :] / temperature
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=-1)

  return idx


def top_k_sampling(model, idx, max_new_tokens, context_size, temperature: float=1.0, k: int=3):
  for _ in range(max_new_tokens):
    segment = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(segment) #dimension [batch_size, context_size, vocab_size]
    logits = logits[:, -1, :] #only need the last word's pmf for each sample in batch
    vals = torch.topk(logits, k, dim=-1).values[:, -1:]
    logits = torch.where( logits >= vals, logits, torch.tensor(float('-inf')).to(logits.device)) / temperature 
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=-1)

  return idx
