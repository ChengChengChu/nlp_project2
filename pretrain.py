import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import join
import math
from argparse import ArgumentParser
from DailyData import Daily
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from chat_load import post_set
from lsp_model.optim import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
from tqdm import tqdm
import re
import json
torch.manual_seed(100)

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main() :
    parser = ArgumentParser()
    parser.add_argument("--emotion", type=str, default="angry")
    # parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--save", type=str, default="model/save/")
    parser.add_argument("--model", type=str, default='microsoft/DialoGPT-small')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--specific", type=str, default=None)
    # parser.add_argument("--ra", type=float, default=3)
    # parser.add_argument("--topic", type=str, default='gender')
    parser.add_argument("--prefix", type=str, default=None)
    # parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    args = parser.parse_args()
    np.random.seed(100)

    model_train = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    os.makedirs('model/' + args.save, exist_ok=True)

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
    

    post = Daily('data/daily_train_key.json', tokenizer, args)
    train_dataloader = DataLoader(post, batch_size=args.batch, shuffle=True, num_workers=1)
    batch = 0
    temp_score = 0
    loss = 0

    for global_step in range(1):
        model_train.train()
        for inputs_id, mask, ll in tqdm(train_dataloader):
            batch += 1
            output = model_train(inputs_id, past_key_values=None, attention_mask=mask, labels=inputs_id)
            # print(output['loss'])
            loss += output['loss']
            if batch % 4 == 0 :
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  
                print('loss : ', loss)
               
                loss = 0
                # break
                torch.save(model_train.state_dict(), 'model/' + args.save + '.pt')

            # break
    

    


    # pass

if __name__ == "__main__" :
    main()