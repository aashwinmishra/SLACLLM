import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from typing import List, Dict, Tuple


def load_data(file_path: str):
  with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
  return text_data


class GPTDataset(torch.utils.data.Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []
    token_ids = tokenizer.encode(txt)

    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i: i+max_length]
      target_chunk = token_ids[i+1: i+1+max_length]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
  dataset = GPTDataset(txt, tokenizer, max_length, stride)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
  return dataloader


