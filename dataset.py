'''
В этом файле определен класс датасета для обучения с помощью перекрестной энтропии
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
        Класс датасета для данных, обработанных с помощью скриптов max_article_sizes.py и prepare_data.py
        Args:
            root_dir: python str: папка с данными в формате json файлов. Например, CNN-DM/stories_tokenized (смотри readme)
            ids_file: python str: путь с json-файлу с разбиением датасета на трейн, валидацию и тест (смотри readme)
            mode: python str: одно из train, valid или test. В зависимости от значения загрузит различные данные
            length: python int или None: максимальное количество данных. Позволяет уменьшить количество данных для ускорения эпох, валидации и тестирования
        '''
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
        '''
        Возвращает словарь с:
        'article' : тензором последовательности токенов исходного текста и резюме, разделенных
        'sum_idx' : индексом токена-разделителя
        'attention_mask' : тензором с маской пэддингов
        '''
        idx = self.files[self.idxs[idx]]
        file_name = os.path.join(self.root_dir, str(idx))
        with open(file_name, 'r') as f:
            data = json.load(f)
        text = self.pad * 1024
        if len(data['abstract']) < 100 or len(data['abstract']) + len(data['article']) < 1023:
            abstract = data['abstract'] + self.tokenizer.encode(self.tokenizer.bos_token)
        else:
            abstract = data['abstract'][:-1] + self.tokenizer.encode(self.tokenizer.bos_token)

        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + abstract
        text[:len(content)] = content
        text = torch.Tensor(text)
        mask = torch.where(text == self.pad[0], torch.zeros_like(text), torch.ones_like(text))
        sample = {'article': text.long(), 'sum_idx': len(data['article']), 'attention_mask': mask.long()}
        return sample
