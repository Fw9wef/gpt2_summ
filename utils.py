import os
import random
import numpy as np
import torch
from transformers import GPT2Tokenizer


def add_special_tokens():
    """ Returns GPT2 tokenizer after adding separator and padding tokens """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(all_args):
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    if all_args.n_gpu > 0:
        torch.cuda.manual_seed_all(all_args.seed)


def make_dirs(all_args):
    for path in [all_args.output_dir, all_args.model_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)


def sample_seq(model, context, length, device, bos_token):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    flag = False
    with torch.no_grad():  
        for _ in range(length):
            if flag:
                break
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token.item() == bos_token:
                flag = True
            generated = torch.cat((generated, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
    return generated


def generate_sample(data, tokenizer, model, num=1, eval_step=False, length=100, device=torch.device('cuda')):
    for i in range(num):
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx+1].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        generated_text = sample_seq(model, context, length, device, tokenizer.bos_token_id)
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
        generated_text = sample_seq(model, context, length, all_args.device, tokenizer.bos_token_id)
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
