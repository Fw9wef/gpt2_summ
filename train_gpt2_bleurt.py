import argparse
from datetime import datetime
import json
import os
import pickle
import random
import time

import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel,AdamW, GPT2Tokenizer, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm

from rl_dataset import GPT21024Dataset
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, top_k_top_p_filtering

import tensorflow as tf
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from bleurt import score


output_dir = './test_out'
num_workers = 4
lr = 5e-5
num_train_epochs = 5
max_grad_norm = 1.
txt_gen_len = 256
batch_size = 2
forward_batch_size = 1
ppo_config = {
    'batch_size': batch_size,
    'forward_batch_size': forward_batch_size
}
metrics_file = './test_out/metrics'


train_dataset = GPT21024Dataset('CNN/gpt2_1024_data', 'CNN/ids.json', mode='train',
                                 length=3000)
val_dataset = GPT21024Dataset('CNN/gpt2_1024_data', 'CNN/ids.json', mode='valid',
                                 length=500)
tokenizer = add_special_tokens()
ignore_idx = tokenizer.pad_token_id

train_sampler = RandomSampler(train_dataset)
val_sampler = RandomSampler(val_dataset)

train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                      num_workers=num_workers)
val_dl = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size*2,
                    num_workers=num_workers)

gpt2_model = GPT2HeadWithValueModel.from_pretrained('./weights/')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('./weights/')

device = torch.device('cuda:0')
gpt2_model = torch.nn.DataParallel(gpt2_model, device_ids=[0, 1, 2])
gpt2_model_ref = torch.nn.DataParallel(gpt2_model_ref, device_ids=[0, 1, 2])
_ = gpt2_model.to(device)
_ = gpt2_model_ref.to(device)

with tf.device('gpu:3'):
    scorer = score.BleurtScorer('../bleurt/bleurt/bleurt-base-512')



for epoch in range(10):
    print("Starting epoch: %d" % (epoch + 1))
    step_reward = []
    epoch_iterator = tqdm(train_dl, desc="Training")
    for step, batch in enumerate(epoch_iterator):
        torch.cuda.empty_cache()
        query, summary = batch['article'].to(device), batch['summary']

        response_tensors = []
        for i in range(int(tbs / fbs)):
            response = respond_to_batch(gpt2_model, query[i * fbs:(i + 1) * fbs], txt_len=256)
            response_tensors.append(response)
        response_tensors = torch.cat(response_tensors)

        response_data = [tokenizer.decode(response_tensors[i, :]) for i in range(tbs)]
        reference_data = [tokenizer.decode(summary[i, :]) for i in range(tbs)]

        rewards = scorer.score(reference_data, response_data)
        mean_step_reward = np.mean(rewards)
        step_reward.append(mean_step_reward)
        rewards = torch.tensor(rewards).to(device)

        stats = ppo_trainer.step(query, response_tensors, rewards)
    print("Mean train reward: %f" % np.mean(step_reward))
    train_reward.append(step_reward)

    val_step_reward = []
    val_iterator = tqdm(val_dl, desc="Validating")
    for step, batch in enumerate(epoch_iterator):
        torch.cuda.empty_cache()
        with torch.no_grads():
            query, summary = batch['article'].to(device), batch['summary']
            response_tensors = []
            for i in range(int(tbs / fbs)):
                response = respond_to_batch(gpt2_model, query[i * fbs:(i + 1) * fbs], txt_len=256)
                response_tensors.append(response)
            response_tensors = torch.cat(response_tensors)

            response_data = [tokenizer.decode(response_tensors[i, :]) for i in range(tbs)]
            reference_data = [tokenizer.decode(summary[i, :]) for i in range(tbs)]

            rewards = scorer.score(reference_data, response_data)
            mean_step_reward = np.mean(rewards)
            val_step_reward.append(mean_step_reward)
    print("Mean validation reward: %f" % np.mean(val_step_reward))
    val_reward.append(val_step_reward)