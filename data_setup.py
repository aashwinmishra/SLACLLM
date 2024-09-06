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


