import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from tqdm import tnrange


def add_special_tokens():
	""" Returns GPT2 tokenizer after adding separator and padding tokens """
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
	num_add_toks = tokenizer.add_special_tokens(special_tokens)
	return tokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0, bos_token=50256):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    flag = False
    with torch.no_grad():  
        for _ in range(length):
            if flag:
                break
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            # next_token_logits = outputs[0][0, -1, :] / temperature
            # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            next_token_logits = outputs[0][0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token.item() == bos_token:
                flag = True
            generated = torch.cat((generated, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
    return generated


def beam_search(model, context, length, beam_size, device, temperature=1):
    """ Generate sequence using beam search https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            beam_size: >=1 and <= total_no_of_tokens
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    with torch.no_grad():  
        inputs = {'input_ids': context}
        outputs = model(**inputs) 
        next_token_logits = outputs[0][0, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits)
        scores, indices = torch.topk(next_token_probs, beam_size)
        indices = indices.tolist()
        sequences = [[c] for c in indices]
        for _ in range(length-1):
            logits = torch.zeros(beam_size*len(next_token_logits))
            for j in range(len(sequences)):
                new_generated = torch.cat((context,torch.tensor([sequences[j]], dtype=torch.long, device=device)),dim=1)
                inputs = {'input_ids': new_generated}
                outputs = model(**inputs) 
                next_token_logits = outputs[0][0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits)
                start, stop = j*len(next_token_logits), (j+1)*len(next_token_logits)
                logits[start:stop] = scores[j]*next_token_probs
            scores, new_logits_indices = torch.topk(logits,beam_size)
            logits = (new_logits_indices%50259).tolist()
            for j in range(len(sequences)):
                sequences[j] = sequences[j]+[logits[j]]
    return scores, sequences


def generate_beam_sample(data, tokenizer, model, num=1, length=100, beam_size=3, device=torch.device('cuda')):
    """ Generate summaries for "num" number of articles using beam search.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            num = number of articles for which summaries has to be generated
    """
    for i in range(num):
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx+1].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        scores, sequences = beam_search(model, context, length, beam_size, device)
        print('new_article', end='\n\n')
        print(tokenizer.decode(context[:-1]), end='\n\n')
        print('actual_summary', end='\n\n')
        print(tokenizer.decode(summary), end='\n\n')
        for i in range(len(sequences)):
            text = tokenizer.convert_ids_to_tokens(sequences[i],skip_special_tokens=True)
            text = tokenizer.convert_tokens_to_string(text)  
            print("generated_summary-{} and Score is {}.".format(i+1, scores[i]), end='\n\n')
            print(text, end='\n\n')


def generate_sample(data, tokenizer, model, num=1, eval_step=False, length=100, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda')):
    """ Generate summaries for "num" number of articles.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
            num = number of articles for which summaries has to be generated
            eval_step = can be True/False, checks generating during evaluation or not
    """
    for i in range(num):
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx+1].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p,
                                    bos_token=tokenizer.encode(tokenizer.bos_token))
        generated_text = generated_text[0, len(context):].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        if eval_step==False:
            print('new_article', end='\n\n')
            print(tokenizer.decode(context, skip_special_tokens=True), end='\n\n')
            print("generated_summary", end='\n\n')
            print(text, end='\n\n')
            print('actual_summary', end='\n\n')
            print(tokenizer.decode(summary, skip_special_tokens=True), end='\n\n')
        else:
            print(tokenizer.decode(generated_text, skip_special_tokens=True), end='\n\n')
            print("generated_summary", end='\n\n')


def watch_metrics(all_args, model, tokenizer, data, num=100, mode='train', length=100):
    if num > len(data):
        num = len(data)

    bleurt_s = []
    r1_s, r2_s, rl_s = [], [], []
    for i in np.random.choice(len(data), num, replace=False):
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx+1].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        generated_text = sample_seq(model, context, length, device=all_args.device,
                                    bos_token=tokenizer.encode(tokenizer.bos_token))
        generated_text = generated_text[0, len(context):].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        metrics_dict = calc_metrics(tokenizer.decode(summary, skip_special_tokens=True), text)

        bleurt_s.append(metrics_dict['bleurt'])
        r1_s.append(metrics_dict['r1'])
        r2_s.append(metrics_dict['r2'])
        rl_s.append(metrics_dict['rl'])

    if mode == 'train':
        path_to_metrics = os.path.join(all_args.output_dir, 'train_')
    elif mode == 'val':
        path_to_metrics = os.path.join(all_args.output_dir, 'val_')
    else:
        path_to_metrics = os.path.join(all_args.output_dir, 'test_')

    with open(path_to_metrics+'bleurt.txt', 'a') as f:
        f.write("%.6f\n" % np.mean(bleurt_s))
    with open(path_to_metrics+'r1.txt', 'a') as f:
        f.write("%.6f\n" % np.mean(r1_s))
    with open(path_to_metrics+'r2.txt', 'a') as f:
        f.write("%.6f\n" % np.mean(r2_s))
    with open(path_to_metrics+'rl.txt', 'a') as f:
        f.write("%.6f\n" % np.mean(rl_s))


class SaveModelDataParallel(torch.nn.DataParallel):
    def __getattr__(self, item):
        if item == 'save_pretrained':
            return getattr(self.module, item)
        else:
            return super().__getattr__(item)


import tensorflow as tf
tf.compat.v1.flags.DEFINE_integer('batch_size', 1, 'batch_size')
tf.compat.v1.flags.DEFINE_float("lr", 5e-5, "learning rate")
tf.compat.v1.flags.DEFINE_integer("seed", 42, "seed to replicate results")
tf.compat.v1.flags.DEFINE_integer("n_gpu", 1, "no of gpu available")
tf.compat.v1.flags.DEFINE_integer("gradient_accumulation_steps", 32, "gradient_accumulation_steps")
tf.compat.v1.flags.DEFINE_integer("num_workers", 4, "num of cpus available")
tf.compat.v1.flags.DEFINE_integer("device", -1, "torch.device object")
tf.compat.v1.flags.DEFINE_integer("num_train_epochs", 5, "no of epochs of training")
tf.compat.v1.flags.DEFINE_string("output_dir", './output', "path to save evaluation results")
tf.compat.v1.flags.DEFINE_string("model_dir", './weights', "path to save trained model")
tf.compat.v1.flags.DEFINE_float("max_grad_norm", 1.0, "max gradient norm.")
tf.compat.v1.flags.DEFINE_string("root_dir", './CNN-DM/gpt2_1024_data', "location of json dataset.")
tf.compat.v1.flags.DEFINE_string("ids_file", './CNN-DM/ids.json', "location of train, valid and test file indexes")
from bleurt import score
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
with tf.device('cpu'):
    bleurt_scorer = score.BleurtScorer('../bleurt/bleurt/bleurt-base-512')

def calc_metrics(reference, candidate):
    r_scores = rouge_scorer.score(reference, candidate)
    b_score = bleurt_scorer.score([reference], [candidate], batch_size=1)
    metrics = {'r1': r_scores['rouge1'][2],
               'r2': r_scores['rouge2'][2],
               'rl': r_scores['rougeL'][2],
               'bleurt': b_score[0]}
    return metrics
