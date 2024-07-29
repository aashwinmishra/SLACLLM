import torch
import torch.nn as nn


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


