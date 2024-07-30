import torch
import torch.nn as nn
import tiktoken


def calc_loss_batch(input_batch: torch.tensor, 
                    target_batch: torch.tensor, 
                    model: torch.nn.Module, 
                    device: torch.device):
  input_batch, target_batch = input_batch.to(device), target_batch.to(device)
  preds = model(input_batch)
  loss = torch.nn.functional.cross_entropy(preds.flatten(0,1), target_batch.flatten())
  return loss


def calc_loss_loader(data_loader: torch.utils.data.DataLoader, 
                     model: torch.nn.Module, 
                     device: torch.device, 
                     num_batches: int=None):
  total_loss = 0.0
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(len(data_loader), num_batches)

  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    else:
      break
  return total_loss/num_batches


def evaluate_model(model: torch.nn.Module, 
                   train_loader: torch.utils.data.DataLoader, 
                   val_loader: torch.utils.data.DataLoader, 
                   device: torch.device, 
                   eval_iter: int):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
  model.eval()
  return train_loss, val_loss


def generate_and_print_sample(model: torch.nn.Module, 
                              tokenizer, 
                              device: torch.device, 
                              start_context: str):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_text_simple(model, encoded, 50, context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
  model.train()


def train_model_simple(model: torch.nn.Module,
                       train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device,
                       num_epochs: int,
                       eval_freq: int,
                       eval_iter: int,
                       start_context: int,
                       tokenizer):
  train_losses, val_losses, track_tokens_seen = [], [], []
  tokens_seen, global_step = 0, -1

  for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      loss.backward()
      optimizer.step()
      tokens_seen += input_batch.numel()
      global_step += 1

      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    generate_and_print_sample(model, tokenizer, device, start_context)

  return train_losses, val_losses, track_tokens_seen


