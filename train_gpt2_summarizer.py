import argparse
from datetime import datetime
import json
import os
import pickle
import random
import time

import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, GPT2Tokenizer, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm

from dataset import GPT21024Dataset
from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, \
    top_k_top_p_filtering, SaveModelDataParallel, watch_metrics


def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
    """ Trains GPT2 model and logs necessary details.
            Args:
                args: dict that contains all the necessary information passed by user while training
                 model: finetuned gpt/gpt2 model
                tokenizer: GPT/GPT2 tokenizer
                train_dataset: GPT21024Dataset object for training data
                ignore_index: token not considered in loss calculation
        """
    writer = SummaryWriter('./logs')
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                          num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)
    for epoch_number in range(1, args.num_train_epochs+1):
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            attention_mask = torch.tensor(batch['attention_mask']).to(args.device)
            model.train()
            logits = model(inputs, attention_mask=attention_mask)[0]
            #logits = model(inputs)[0]
            index = batch['sum_idx']  # index of separator token
            # only consider loss on reference summary just like seq2seq models
            loss = 0
            for idx, logs, labs in zip(index, logits, labels):
                shift_logits = logs[..., idx:-1, :].contiguous()
                shift_labels = labs[..., idx + 1:].contiguous()
                loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / args.gradient_accumulation_steps / index.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            #if (step + 1) % args.gradient_accumulation_steps == 0:
            if True:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss) / args.gradient_accumulation_steps, global_step)
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                if (step + 1) / args.gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=False, device=args.device)
                    watch_metrics(args, model, tokenizer, train_dataset, num=100, mode='train')

            #if (step + 1) % (50 * args.gradient_accumulation_steps) == 0:
            if True:
                results = evaluate(args, model, valid_dataset, ignore_index, global_step)
                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)
                print('After', global_step + 1, 'updates: ', end='\n\n')
                generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=True, device=args.device)
                watch_metrics(args, model, tokenizer, valid_dataset, num=100, mode='val')
            break

        new_model_dir = os.path.join(args.model_dir, str(epoch_number))
        os.mkdir(new_model_dir)
        model.save_pretrained(new_model_dir)


def evaluate(args, model, eval_dataset, ignore_index, global_step=None):
    """Returns perplexity score on validation dataset.
          Args:
              args: dict that contains all the necessary information passed by user while training
              model: finetuned gpt/gpt2 model
              eval_dataset: GPT21024Dataset object for validation data
              global_step: no. of times gradients have backpropagated
              ignore_index: token not considered in loss calculation
    """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['article']).to(args.device), torch.tensor(batch['article']).to(args.device)
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            logits = model(inputs, attention_mask=attention_mask)[0]
            index = batch['sum_idx']  # index of separator token
            # only consider loss on reference summary just like seq2seq models
            for idx, logs, labs in zip(index, logits, labels):
                shift_logits = logs[..., idx:-1, :].contiguous()
                shift_labels = labs[..., idx + 1:].contiguous()
                lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                eval_loss += lm_loss.item()
                nb_eval_steps += 1
        break

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--n_gpu", default=1, type=int, required=False, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int, required=True,
                        help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", default=1, type=int, required=True, help="batch_size")
    parser.add_argument("--num_workers", default=4, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device", default=-1, required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs", default=5, type=int, required=True, help="no of epochs of training")
    parser.add_argument("--output_dir", default='./output', type=str, required=True,
                        help="path to save evaluation results")
    parser.add_argument("--model_dir", default='./weights', type=str, required=True, help="path to save trained model")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--root_dir", default='./CNN-DM/gpt2_1024_data', type=str, help="location of json dataset.")
    parser.add_argument("--ids_file", default='./CNN-DM/ids.json', type=str,
                        help="location of train, valid and test file indexes")
    # arser.add_argument("--rl_mode", action='store_true', help="if specified? trains model with reinforcement learning approach")
    args = parser.parse_args()

    train_data = GPT21024Dataset(args.root_dir, args.ids_file, mode='train')
    valid_data = GPT21024Dataset(args.root_dir, args.ids_file, mode='valid', length=500)
    tokenizer = add_special_tokens()
    ignore_idx = tokenizer.pad_token_id

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    if args.device == -1:
        model = SaveModelDataParallel(model, device_ids=[0, 1, 2, 3])
    args.device = torch.device('cuda:0')
    model.to(args.device)

    start = time.time()
    train(args, model, tokenizer, train_data, valid_data, ignore_idx)
    print('total time: ', (time.time() - start) / 60, " minutes", end='\n\n')

    print('Saving trained model...')
    #model.save_pretrained(args.model_dir)


if __name__ == '__main__':
    main()
