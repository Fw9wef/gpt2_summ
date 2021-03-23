import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from rl_dataset import GPT21024Dataset
from utils import add_special_tokens, SaveModelDataParallel

import tensorflow as tf
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from bleurt import score
from rouge_score import rouge_scorer


output_dir = './trl_output'
num_workers = 2
lr = 5e-5
num_train_epochs = 5
max_grad_norm = 1.
txt_gen_len = 100
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
batch_size = ppo_config['batch_size']


train_dataset = GPT21024Dataset('CNN/gpt2_1024_data', 'CNN/ids.json', mode='train', length=3000)
val_dataset = GPT21024Dataset('CNN/gpt2_1024_data', 'CNN/ids.json', mode='valid', length=500)
tokenizer = add_special_tokens()
ignore_idx = tokenizer.pad_token_id

train_sampler = RandomSampler(train_dataset)
train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
val_sampler = RandomSampler(val_dataset)
val_dl = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers)


gpt2_model = GPT2HeadWithValueModel.from_pretrained('./weights/partial_masked/')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('./weights/partial_masked/')
device = torch.device('cuda:0')
#gpt2_model = torch.nn.SaveModelDataParallel(gpt2_model, device_ids=[0, 1, 2])
#gpt2_model_ref = torch.nn.SaveModelDataParallel(gpt2_model_ref)
_ = gpt2_model.to(device)
_ = gpt2_model_ref.to(device)
with tf.device('cpu'):
    scorer = score.BleurtScorer('../bleurt/bleurt/bleurt-base-512/')
r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)
tbs, fbs = ppo_config['batch_size'], ppo_config['forward_batch_size']


def validate():
    val_step_reward = []
    val_step_r1 = []
    val_step_r2 = []
    val_step_rl = []
    val_iterator = tqdm(val_dl, desc="Validating")
    for step, batch in enumerate(val_iterator):
        torch.cuda.empty_cache()
        with torch.no_grad():
            query, summary = batch['article'].to(device), batch['summary']
            response_tensors = []
            for i in range(int(tbs / fbs)):
                response = respond_to_batch(gpt2_model, query[i * fbs:(i + 1) * fbs], txt_len=txt_gen_len)
                response_tensors.append(response)
            response_tensors = torch.cat(response_tensors)

            response_data = [tokenizer.decode(response_tensors[i, :]) for i in range(tbs)]
            reference_data = [tokenizer.decode(summary[i, :]) for i in range(tbs)]

            r1s, r2s, rls = list(), list(), list()
            for a, b in zip(reference_data, response_data):
                scores = r_scorer.score(a, b)
                r1s.append(scores['rouge1'][2])
                r2s.append(scores['rouge2'][2])
                rls.append(scores['rougeL'][2])
            val_step_r1.append(np.mean(r1s))
            val_step_r2.append(np.mean(r2s))
            val_step_rl.append(np.mean(rls))

            rewards = scorer.score(reference_data, response_data)
            mean_step_reward = np.mean(rewards)
            val_step_reward.append(mean_step_reward)

    print("Mean validation reward: %f" % np.mean(val_step_reward))
    with open(os.path.join(output_dir, 'val_r1.txt'), 'a') as f:
        f.write(str(np.mean(val_step_r1)) + '\n')
    with open(os.path.join(output_dir, 'val_r2.txt'), 'a') as f:
        f.write(str(np.mean(val_step_r2)) + '\n')
    with open(os.path.join(output_dir, 'val_rl.txt'), 'a') as f:
        f.write(str(np.mean(val_step_rl)) + '\n')
    with open(os.path.join(output_dir, 'val_bleurt.txt'), 'a') as f:
        f.write(str(np.mean(val_step_reward)) + '\n')


total_steps_passed = 0
for epoch in range(10):
    print("Starting epoch: %d" % (epoch + 1))
    step_reward = []
    step_r1 = []
    step_r2 = []
    step_rl = []
    epoch_iterator = tqdm(train_dl, desc="Training")
    for step, batch in enumerate(epoch_iterator):
        total_steps_passed += 1
        torch.cuda.empty_cache()
        query, summary = batch['article'].to(device), batch['summary']

        response_tensors = []
        for i in range(int(tbs / fbs)):
            response = respond_to_batch(gpt2_model, query[i * fbs:(i + 1) * fbs], txt_len=txt_gen_len)
            response_tensors.append(response)
        response_tensors = torch.cat(response_tensors)

        response_data = [tokenizer.decode(response_tensors[i, :], skip_special_tokens=True) for i in range(tbs)]
        reference_data = [tokenizer.decode(summary[i, :], skip_special_tokens=True) for i in range(tbs)]

        r1s, r2s, rls = list(), list(), list()
        for a, b in zip(reference_data, response_data):
            scores = r_scorer.score(a, b)
            r1s.append(scores['rouge1'][2])
            r2s.append(scores['rouge2'][2])
            rls.append(scores['rougeL'][2])
        step_r1.append(np.mean(r1s))
        step_r2.append(np.mean(r2s))
        step_rl.append(np.mean(rls))

        rewards = scorer.score(reference_data, response_data)
        mean_step_reward = np.mean(rewards)
        step_reward.append(mean_step_reward)
        rewards = torch.tensor(rewards).to(device)

        with open(os.path.join(output_dir, 'train_r1.txt'), 'a') as f:
            f.write(str(step_r1[-1]) + '\n')
        with open(os.path.join(output_dir, 'train_r2.txt'), 'a') as f:
            f.write(str(step_r2[-1]) + '\n')
        with open(os.path.join(output_dir, 'train_rl.txt'), 'a') as f:
            f.write(str(step_rl[-1]) + '\n')
        with open(os.path.join(output_dir, 'train_bleurt.txt'), 'a') as f:
            f.write(str(step_reward[-1]) + '\n')

        stats = ppo_trainer.step(query, response_tensors, rewards)

        if total_steps_passed % 1000 == 0:
            validate()

    print("Mean train reward: %f" % np.mean(step_reward))

    path_to_save = os.path.join(output_dir, 'weights', 'epoch' + str(epoch + 1))
    os.mkdir(path_to_save)
    gpt2_model.save_pretrained(path_to_save)
