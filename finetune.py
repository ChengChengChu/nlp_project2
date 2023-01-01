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



def main() :

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_finetune_args()

    set_seed(args.seed)
    set_finetune(args)

    if args.model in model_map:
        m = model_map[args.model]
    else:
        m = args.model
    
    model_train = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m)
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
        for inputs_id, mask, ll in tqdm(train_dataloader):

            inputs_id, mask = inputs_id.to(device), mask.to(device)
            
            output = model_train(inputs_id, attention_mask=mask, labels=inputs_id)
            loss = output['loss']
            loss.backward()

            if batch % 16 == 0 or batch :
                optimizer.step()
                optimizer.zero_grad()
                with open(f"./pretrain_output/{args.save}/loss.txt", 'a') as f:
                    f.write(f"[ Training loss | epoch: {epoch} | step: {batch/16} ]: {loss}\n")

            batch += 1

        torch.save(
            model_train.state_dict(), 
            f"./pretrain_output/{args.save}/models/{args.model}-{epoch}.pt"
            )


if __name__ == "__main__" :
    main()