'''
Этот файл предоставляет возможность тестирования обученной модели и финального потсчета метрик
'''

import argparse
from transformers import GPT2LMHeadModel
import torch
from dataset import GPT21024Dataset
from utils import add_special_tokens, watch_metrics


def test(all_args, model, tokenizer, dataset):
    model.eval()
    watch_metrics(all_args, model, tokenizer, dataset, num=len(dataset), mode='test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--num_workers", default=4, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device", default=3, required=False, help="torch.device object")
    parser.add_argument("--output_dir", default='./output', type=str, required=True,
                        help="path to save evaluation results")
    parser.add_argument("--model_dir", default='./weights', type=str, required=True, help="path to save trained model")
    parser.add_argument("--root_dir", default='./CNN-DM/gpt2_1024_data', type=str, help="location of json dataset.")
    parser.add_argument("--ids_file", default='./CNN-DM/ids.json', type=str,
                        help="location of train, valid and test file indexes")

    all_args = parser.parse_args()
    dataset = GPT21024Dataset(all_args.root_dir, all_args.ids_file, mode='test')
    tokenizer = add_special_tokens()
    model = GPT2LMHeadModel.from_pretrained(all_args.model_dir)
    all_args.device = torch.device('cuda:'+str(all_args.device))
    model.to(all_args.device)

    test(all_args, model, tokenizer, dataset)


if __name__ == '__main__':
    main()
