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
            idx = self.idxs[-idx - self.len]  # assuming valid and test set of same sizes
        else:
            idx = self.idxs[idx]
        # file_name = os.path.join(self.root_dir,str(idx)+".json")
        file_name = os.path.join(self.root_dir, str(idx))
        with open(file_name, 'r') as f:
            data = json.load(f)

        article = self.tokenizer.encode(self.tokenizer.pad_token) * 768
        content = data['article'][:767] + self.tokenizer.encode(self.tokenizer.sep_token)
        article[len(article) - len(content):] = content
        article = torch.tensor(article)

        summary = self.tokenizer.encode(self.tokenizer.pad_token) * 256
        content = data['abstract'][:256]
        summary[:len(content)] = content
        summary = torch.tensor(summary)

        sample = {'article': article, 'summary': summary}
        return sample
