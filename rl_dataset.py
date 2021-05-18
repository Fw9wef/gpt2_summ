'''
В этом файле определен класс датасета для обучения с помощью rl
'''

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import add_special_tokens


class GPT21024Dataset(Dataset):
    def __init__(self, root_dir, ids_file, mode='train', length=None):
        '''
        root_dir:
        ids_file:
        mode:
        length:
        '''
        self.root_dir = root_dir  # папка с данными в формате json файлов (gpt2_1024_data)
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
        article = self.pad * 924
        abstract = self.pad * 100
        if len(data['abstract']) < 100 or len(data['abstract']) + len(data['article']) < 1023:
            abstract_content = data['abstract'] + self.tokenizer.encode(self.tokenizer.bos_token)
        else:
            abstract_content = data['abstract'][:-1] + self.tokenizer.encode(self.tokenizer.bos_token)
        abstract[:len(abstract_content)] = abstract_content
        abstract = torch.Tensor(abstract)

        article_content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token)
        article[:len(article_content)] = article_content
        article = torch.Tensor(article)
        mask = torch.where(article == self.pad[0], torch.zeros_like(article), torch.ones_like(article))
        sample = {'article': article.long(), 'abstract': abstract.long(), 'article_mask': mask.long()}
        return sample