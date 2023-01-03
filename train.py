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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def set_model(args) :
    if args.model in model_map:
        m = model_map[args.model]
    else :
        m = args.model
    
    model = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m)

    if args.ckpt != None : 
        print("\nUsing model with finetuning !!!\n")
        model.load_state_dict(torch.load(args.ckpt))
    else : 
        print("\nTraining with base model without finetuning !!!\n")
    
    return model, tokenizer

    

def main() :
    analyzer = SentimentIntensityAnalyzer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model_train, tokenizer = set_model(args)
    model_train = model_train.to(device)

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
    
    model_train.train()
    
    for epoch in range(args.epoch):
        batch = 0
        loss = 0
        pbar = tqdm(train_dataloader)

        for inputs_id, mask, length in pbar:
            inputs_id = inputs_id[:, :length].to(device)
            mask = mask[:, :length].to(device)

            output = model_train(inputs_id, past_key_values=None, attention_mask=mask, labels=inputs_id)
            prev_input, past = output['logits'], output['past_key_values']

            eos = [tokenizer.encoder["<|endoftext|>"]]

            prev_input = torch.LongTensor([[eos] * inputs_id.shape[0]]).to(device)

            temp_sentence = [[] for i in range(inputs_id.shape[0])]
            model_train_CrossEntropy = [0 for i in range(inputs_id.shape[0])]

            append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device)
            mask = torch.cat((mask, append), 1)

            for i in range(40): 
                prev_input = prev_input.to(device)
                model_train_out = model_train(prev_input, past_key_values=past)
                logits, past = model_train_out['logits'], model_train_out['past_key_values']
                mask = torch.cat((mask, append), 1)

                logits = logits.squeeze(0).squeeze(1)

                logits = torch.softmax(logits, dim=-1)
                prev_input = torch.multinomial(logits[:], num_samples=1)

                for j in range(inputs_id.shape[0]):
                    if i != 0 and temp_sentence[j][-1] == eos[0]: continue
                    temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
                    model_train_CrossEntropy[j] = temp_loss + model_train_CrossEntropy[j]

                if i == 0:
                    for j in range(inputs_id.shape[0]):
                        temp_sentence[j].append(prev_input[j].item())
                    continue
                flag = 1
                
                for j in range(0, inputs_id.shape[0]):
                    if temp_sentence[j][-1] != eos[0]: 
                        flag = 0
                        temp_sentence[j].append(prev_input[j].item())

                if flag == 1: break
            decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]
            decode_sentence = []

            for s in decode_temp_sentence :
                decode_sentence.append(s.replace('\n', '. '))
                
            reward = []

            for s in decode_sentence : 
                tmp_1, tmp_2 = replace_sentence(s)
                vs_1 = analyzer.polarity_scores(tmp_1)
                vs_2 = analyzer.polarity_scores(tmp_2)
                reward.append(abs(vs_1['compound'] - vs_2['compound']))

            reward = np.array(reward)
            for j in range(inputs_id.shape[0]) :
                loss = loss + model_train_CrossEntropy[j] * reward[j] 

            #### calculate loss
            loss.backward()
            if batch % 16 == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})
        
            batch += 1






    


if __name__ == "__main__" :
    args = get_train_args()
    main()

