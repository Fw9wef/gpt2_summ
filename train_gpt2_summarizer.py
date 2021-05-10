'''
В этом файле находятся циклы обучения и валидации gpt2 с помощью перекресной энтропии.
'''

import argparse
import os
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from dataset import GPT21024Dataset
from utils import add_special_tokens, generate_sample, set_seed, SaveModelDataParallel, watch_metrics


def train(all_args, model, tokenizer, train_dataset, valid_dataset, ignore_index):
    """
    Trains GPT2 model and logs necessary details.
        Args:
            all_args: argparse object with train loop settings
            model: gpt2 model
            tokenizer: GPT2 tokenizer
            train_dataset: GPT21024Dataset object with training data
            ignore_index: token not considered in loss calculation
    """

    # создаем даталоадер из трейнового датасета
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=all_args.batch_size,
                          num_workers=all_args.num_workers)
    # задаем лосс функцию, оптимизатор и планировщик
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation
    optimizer = AdamW(model.parameters(), lr=all_args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, 80000)

    global_step = 0
    model.zero_grad()
    set_seed(all_args)
    for epoch_number in range(1, all_args.num_train_epochs + 1):
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = torch.tensor(batch['article']), torch.tensor(batch['article'])
            inputs = inputs.to(all_args.device)
            labels = labels.to(all_args.device)
            attention_mask = torch.tensor(batch['attention_mask']).to(all_args.device)
            model.train()
            logits = model(inputs, attention_mask=attention_mask)[0]
            index = batch['sum_idx']    # тут индексы токенов-разделителей для каждой последовательности в батче shape = (batch,)
            loss = 0
            for idx, logs, labs in zip(index, logits, labels):
                shift_logits = logs[idx:-1, :]  # для вычисления лоса берем часть последовательностей справа от сепаратора
                shift_labels = labs[idx + 1:]   # смещаем предсказания и лейблы на одну позицию относительно друг друга
                loss += loss_fct(shift_logits, shift_labels)
            loss = loss / all_args.gradient_accumulation_steps / index.shape[0]  # лосс делится на размер батча
                                                                            # и количество шагов аккамуляции градиента
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), all_args.max_grad_norm)
            if (step + 1) % all_args.gradient_accumulation_steps == 0:
                optimizer.step()    # делаем шаг оптимизатора раз в gradient_accumulation_steps батчей
                scheduler.step()
                model.zero_grad()
                global_step += 1
                print("updates passed: %d\tloss: %f" % (global_step, loss.item()), end='\n\n')

            if (step + 1) % (500 * all_args.gradient_accumulation_steps) == 0:
                # раз в 30 шагов оптимизатора выводим сэмплы резюме и подсчитываем метрики на валидации
                print('After', global_step + 1, 'updates: ', end='\n\n')
                evaluate(all_args, model, valid_dataset, ignore_index)
                generate_sample(valid_dataset, tokenizer, model, num=2, eval_step=True, device=all_args.device)
                watch_metrics(all_args, model, tokenizer, valid_dataset, num=50, mode='val')

        # сохраняем обученную модель каждую эпоху
        new_model_dir = os.path.join(all_args.model_dir, str(epoch_number))
        os.mkdir(new_model_dir)
        model.save_pretrained(new_model_dir)


def evaluate(all_args, model, eval_dataset, ignore_index):
    """
    Returns perplexity score on validation dataset.
        Args:
            all_args: dict that contains all the necessary information passed by user while training
            model: gpt2 model
            eval_dataset: GPT21024Dataset object for validation data
            global_step: no. of times gradients have backpropagated
            ignore_index: token not considered in loss calculation
    """
    # далее все аналогично функции train, но без бэкпропа
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=all_args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = torch.tensor(batch['article']).to(all_args.device), torch.tensor(batch['article']).to(all_args.device)
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            logits = model(inputs, attention_mask=attention_mask)[0]
            index = batch['sum_idx']
            for idx, logs, labs in zip(index, logits, labels):
                shift_logits = logs[idx:-1, :]
                shift_labels = labs[idx + 1:]
                lm_loss = loss_fct(shift_logits, shift_labels)
                eval_loss += lm_loss.item()
                nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    with open(os.path.join(all_args.output_dir, 'val_perplexity.txt'), 'a') as f:
        f.write("%.6f\n" % perplexity.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, required=False, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed to replicate results")
    parser.add_argument("--n_gpu", default=1, type=int, required=False, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int, required=True,
                        help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", default=1, type=int, required=True, help="batch_size")
    parser.add_argument("--num_workers", default=2, type=int, required=False, help="num of cpus available")
    parser.add_argument("--device", default=0, required=False, help="torch.device object")
    parser.add_argument("--num_train_epochs", default=5, type=int, required=True, help="no of epochs of training")
    parser.add_argument("--output_dir", default='./output', type=str, required=True,
                        help="path to save evaluation results")
    parser.add_argument("--model_dir", default='./weights', type=str, required=True, help="path to save trained model")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--root_dir", default='./CNN-DM/gpt2_1024_data', type=str, help="location of json dataset.")
    parser.add_argument("--ids_file", default='./CNN-DM/ids.json', type=str,
                        help="location of train, valid and test file indexes")
    all_args = parser.parse_args()

    # загружаем трейновый и валидационный датасеты, текенизатор
    train_data = GPT21024Dataset(all_args.root_dir, all_args.ids_file, mode='train')
    valid_data = GPT21024Dataset(all_args.root_dir, all_args.ids_file, mode='valid', length=500)
    tokenizer = add_special_tokens()
    ignore_idx = tokenizer.pad_token_id

    # загружаем gpt2-small
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    if all_args.n_gpu > 1:
        model = SaveModelDataParallel(model, device_ids=[i for i in range(all_args.n_gpu)])
    all_args.device = torch.device('cuda:' + str(all_args.device))
    model.to(all_args.device)

    train(all_args, model, tokenizer, train_data, valid_data, ignore_idx)


if __name__ == '__main__':
    main()
