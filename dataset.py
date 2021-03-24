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
        self.files = np.sort([x for x in os.listdir(root_dir) if x.endswith('.json')])
        self.mode = mode
        with open(ids_file, 'r') as f:
            self.data = json.load(f)
        if mode == 'train':
            self.idxs = self.data['train_ids']
        elif mode == 'valid':
            self.idxs = self.data['valid_ids']
        else:
            self.idxs = self.data['test_ids']

        if length == None:
            self.len = len(self.idxs)
        else:
            self.len = length


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        idx = self.files[self.idxs[idx]]
        file_name = os.path.join(self.root_dir, str(idx))
        with open(file_name, 'r') as f:
            data = json.load(f)
        text = self.pad * 1024
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        mask = torch.where(text == self.pad[0], 0, 1)
        sample = {'article': text, 'sum_idx': len(data['article']), 'attention_mask': mask}
        return sample
