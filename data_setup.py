import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import urllib.request
import zipfile
import os
from pathlib import Path
from typing import List, Dict, Tuple


class InstructionDataset(torch.utils.data.Dataset):
  def __init__(self, data, tokenizer):
    self.data = data
    self.encoded_texts = []
    for entry in data:
      instruction_plus_entry = format_input(entry)
      response_text = f"\n\n### Response:\n{entry['output']}"
      full_text = instruction_plus_entry + response_text
      self.encoded_texts.append(tokenizer.encode(full_text))

  def __getitem__(self, idx):
    return self.encoded_texts[idx]

  def __len__(self):
    return len(self.data)


def custom_collate(batch, 
                   pad_token_id: int=50256, 
                   ignore_index = -100,
                   allowed_max_length = None,
                   device="cpu"):
  batch_max_length = max(len(item)+1 for item in batch)
  inputs_list, targets_list = [], []

  for item in batch:
    new_item = item.copy()
    new_item += [pad_token_id]
    padded = new_item + [pad_token_id]*(batch_max_length - len(new_item))
    inputs = torch.tensor(padded[:-1])
    targets = torch.tensor(padded[1:])
    inputs_list.append(inputs)
    targets_list.append(targets)

    mask = targets == pad_token_id
    indices = torch.nonzero(mask).squeeze()
    if indices.numel() > 1:
      targets[indices[1:]] = ignore_index

    if allowed_max_length is not None:
      inputs = inputs[:allowed_max_length]
      targets = targets[:allowed_max_length]

  inputs_tensor = torch.stack(inputs_list).to(device)
  targets_tensor = torch.stack(targets_list).to(device)
  return inputs_tensor, targets_tensor


def get_spam_data(url: str="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", 
                  zip_path: str="sms_spam_collection.zip", 
                  extracted_path: str="sms_spam_collection", 
                  data_file_path: str="SMSSpamCollection.tsv"):
  """
  Downloads dataset for Classification finetuning.
  Args:
    url: Location of zipfile with text
    zip_path: Location to download zipfile
    extracted_path: Drive to extract the zipfile contents
    data_file_path: Filename to store data, located in extracted_path
  """
  data_file_path = Path(extracted_path)/data_file_path
  if data_file_path.exists():
    print(f"Data file {data_file_path} exists. Exiting...")
    return 

  with urllib.request.urlopen(url) as response:
    with open(zip_path, "wb") as out_file:
      out_file.write(response.read())

  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extracted_path)

  original_file_path = Path(extracted_path) / "SMSSpamCollection"
  os.rename(original_file_path, data_file_path) #C
  print(f"File downloaded and saved as {data_file_path}")


class SpamDataset(torch.utils.data.Dataset):
  """
  Dataset class for Spam dataset, tokenizes and encodes test. Pads/Truncates all sequences uniformly.
  """
  def __init__(self, csv_file, tokenizer, max_length=None, pad_token=50256):
    self.data = pd.read_csv(csv_file)
    self.encoded_texts = [tokenizer.encode(txt) for txt in self.data["Text"]]
    if max_length is None:
      self.max_length = self._longest_encoded_length()
    else:
      self.max_length = max_length 

    self.encoded_texts = [seq[:self.max_length] for seq in self.encoded_texts]
    self.encoded_texts = [encoded_text + [pad_token]*(self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

  def __getitem__(self, idx):
    encoded = self.encoded_texts[idx]
    label = self.data.iloc[idx]["Label"]
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

  def __len__(self):
    return len(self.data)

  def _longest_encoded_length(self):
    default = 0
    max_length = max([len(seq) for seq in self.encoded_texts])
    return max(default, max_length)


def create_spam_dataloaders(train_path: str, 
                            val_path: str, 
                            test_path: str, 
                            max_length: int=None, 
                            batch_size: int=None):
  """
  Utility function to take paths of the train, validation and test data, create 
  datasets off of the SpamDataset class & return corresponding dataloaders.
  Args:
    train_path: Path for the train data csv file, asssumed to have text, class int entries.
    val_path: Path for the validation data csv file
    test_path: Path for the test data csv file
    max_length: Max length of the sequences, tuncates/pads sequences accordingly.
    batch_size: Batch size for the DataLoader.
  Returns:
    Tuple with train_dataloader, val_dataloader, test_dataloader
  """
  tokenizer = tiktoken.get_encoding('gpt2')
  train_dataset = SpamDataset(csv_file=train_path,
                              max_length=max_length,
                              tokenizer=tokenizer)
  val_dataset = SpamDataset(csv_file=val_path,
                            tokenizer=tokenizer,
                            max_length=train_dataset.max_length)
  test_dataset = SpamDataset(csv_file=test_path,
                             tokenizer=tokenizer,
                             max_length=train_dataset.max_length)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=True, 
                                                 drop_last=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 drop_last=True)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 drop_last=True)
  return train_dataloader, val_dataloader, test_dataloader
                    


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


