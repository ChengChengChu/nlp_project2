import numpy as np
from torch.utils.data import DataLoader
from DailyData import Daily
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
import torch
import torch.nn.functional as F
from lsp_model.optim import Adam
from tqdm import tqdm
from utils import *

def set_model(args):

    if args.model in model_map:
        m = model_map[args.model]
    else:
        m = args.model
    
    model = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m)

    args.ckpt = "pretrain_output/gpt_pretrain/models/gpt-0.pt"

    if args.ckpt != None:
        model.load_state_dict(torch.load(args.ckpt))
    
    return model, tokenizer


def test_finetune(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model, tokenizer = set_model(args)

    prompt = args.prompt

    sentence = generate(model, tokenizer, prompt, device)

    print(sentence)




def train(args) :

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    set_finetune(args)
    model_train, tokenizer = set_model(args)

    model_train.to(device)
    param_optimizer = list(model_train.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, 5e-6,
                     max_grad_norm=1.0)
    

    daily_data = Daily('data/daily_train_key.json', tokenizer, args)
    train_dataloader = DataLoader(daily_data, batch_size=args.batch, shuffle=True, num_workers=4)
    
    
    
    # training
    model_train.train()
    
    for epoch in range(args.epoch):
        batch = 0
        loss = 0
        pbar = tqdm(train_dataloader)
        for inputs_id, mask, length in pbar:

            inputs_id = inputs_id[:, :length].to(device)
            
            # print(inputs_id)
            # print(mask)
            # print("+" * 100)
            
            output = model_train(inputs_id, labels=inputs_id)
            loss = output['loss']
            loss.backward()

            if batch % 16 == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})
                with open(f"./pretrain_output/{args.save}/loss.txt", 'a') as f:
                    f.write(f"[Epoch: {epoch} | step: {batch/16}]: {loss}\n")

            batch += 1

        torch.save(
            model_train.state_dict(), 
            f"./pretrain_output/{args.save}/models/{args.model}-{epoch}.pt"
            )


if __name__ == "__main__" :
    args = get_finetune_args()

    if args.mode == 'train':
        train(args)
    else:
        test_finetune(args)