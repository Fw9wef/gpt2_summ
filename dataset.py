import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from utils import add_special_tokens


class GPT21024Dataset(Dataset):
    def __init__(self, root_dir, ids_file, mode='train', length=None):
        self.root_dir = root_dir
        self.tokenizer = add_special_tokens()
        self.pad = self.tokenizer.encode(self.tokenizer.pad_token)
        print(self.pad, type(self.pad))
        self.idxs = os.listdir(root_dir)
        self.mode = mode
        if len == None:
            self.len = len(self.idxs)
        else:
            self.len = length


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        if self.mode == 'valid':
            idx = self.idxs[-idx]
        elif self.mode == 'test':
            idx = self.idxs[-idx - self.len]
        else:
            idx = self.idxs[idx]
        file_name = os.path.join(self.root_dir, str(idx))
        with open(file_name, 'r') as f:
            data = json.load(f)
        text = self.pad * 1024
        mask = text
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        mask = torch.where(text == self.pad, 1, 0) #
        sample = {'article': text, 'sum_idx': len(data['article']), 'attention_mask': mask}
        return sample