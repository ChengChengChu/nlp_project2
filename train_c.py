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
from decoding import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import wandb

def set_wandb(args):
    wandb.init(
        project="bias_phase1",
        name=f"{args.save}",
        entity="chatbot_ntu"
    )
    wandb.config.update(args)


def set_model(args, name="gen") :

    if name == "inter":
        m = model_map["diologpt"]
    else:
        if args.model in model_map:
            m = model_map[args.model]
        else :
            m = args.model
    
    model = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m)

    if name == "gen":
        if args.ckpt != None: 
            print("Using model with finetuning !!!")
            model.load_state_dict(torch.load(args.ckpt))
        else : 
            print("Training with base model without finetuning !!!")
    
    return model, tokenizer




def make_reward(model, tokenizer, first_input, analyzer, device):
    with torch.no_grad():
        sentences = []
        for i in range(len(first_input)):

            sentences.append((list(first_input[i])))
        m = []
        for i in range(len(sentences)):
            temp_m = [1 for x in range(len(sentences[i]))]
            m.append(temp_m[:])
        eos = [tokenizer.encoder["<|endoftext|>"]]


        # prepare original input to model
        prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in sentences], value=0)).to(device)
        m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0)).to(device)

        position_ids = m.long().cumsum(-1) - 1 #+ prev_input.shape[1]
        position_ids.masked_fill_(m == 0, 1)

        outputs = model(prev_input, past=None, attention_mask=m, position_ids=position_ids)
        past = outputs['past_key_value']

        prev_input = torch.LongTensor([[eos] * len(sentences)]).squeeze(0).to(device)
        append = torch.tensor([[1] for i in range(len(sentences))]).to(device)
        position_ids = m.long().cumsum(-1) - 1
        position_ids.masked_fill_(m == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
        temp_sen = [[] for i in range(len(sentences))]
        
        

        for i in range(60):
            outputs = model(prev_input, past_key_values=past, attention_mask=m, position_ids=position_ids)
            prev_input, past = outputs["logits"], outputs["past_key_values"]
            m = torch.cat((m, append), 1)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

            prev_input = prev_input.squeeze(0).squeeze(1)
            prev_input = top_k_top_p_filtering(prev_input, top_k=40, temperature=.7)
            prev_input = torch.multinomial(prev_input, num_samples=1)

            if i == 0:
                for j in range(len(sentences)):    
                    temp_sen[j].append(prev_input[j].item())
                continue
            flag = 1
            for j in range(len(sentences)):
                if temp_sen[j][-1] != eos[0]: 
                    flag = 0
                    temp_sen[j].append(prev_input[j].item())
            if flag == 1: break
        a = []
        for x in temp_sen:
          a.append(tokenizer.decode(x[:], skip_special_tokens=True).replace('<|endoftext|>', ''))


        vs_1 = analyzer.polarity_scores(a[0])
        vs_2 = analyzer.polarity_scores(a[1])
        return abs(vs_1['compound'] - vs_2['compound'])


def main(args) :
    
    analyzer = SentimentIntensityAnalyzer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    set_train(args)
    set_wandb(args)

    model_train, tokenizer = set_model(args, "gen")
    model_train = model_train.to(device)

    model_inter, tokenizer_inter = set_model(args, "inter")
    model_inter = model_inter.to(device)

    param_optimizer = list(model_train.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = Adam(optimizer_grouped_parameters, args.lr,
                     max_grad_norm=1.0)
    
    daily_data = Daily('data/daily_train_key.json', tokenizer, args)
    train_dataloader = DataLoader(daily_data, batch_size=args.batch, shuffle=True, num_workers=4)
    
    model_train.train()
    model_inter.eval()
    f = open(f"training_output/{args.save}/log.txt", "w")
    count = 0
    for epoch in range(args.epoch):
        batch = 0
        loss = 0
        pbar = tqdm(train_dataloader)
        batch_loss = 0
        batch_reward = 0

        for inputs_id, mask, length in pbar:
            inputs_id = inputs_id[:,0].to(device)
            mask = mask[:,0].to(device)
            eos = [tokenizer.encoder["<|endoftext|>"]]

            prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in inputs_id], value=0)).to(device)
            m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in mask], value=0)).to(device)

            position_ids = m.long().cumsum(-1) - 1 #+ prev_input.shape[1]
            position_ids.masked_fill_(m == 0, 1)
            
            append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

            temp_sentence = [[] for i in range(inputs_id.shape[0])]
            model_train_CrossEntropy = [0 for i in range(inputs_id.shape[0])]

            for j in range(len(inputs_id)):
                temp_sentence[j].extend(inputs_id[j])
            past = None

            for i in range(40): 
                model_train_out = model_train(prev_input, past_key_values=past)
                logits, past = model_train_out['logits'], model_train_out['past_key_values']
                mask = torch.cat((mask, append), 1)
                position_ids = m.long().cumsum(-1) - 1
                position_ids.masked_fill_(m == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                logits = logits.squeeze(0).squeeze(1)
                logits = top_k_top_p_filtering(logits, top_k=40, temperature=.7)
                prev_input = torch.multinomial(logits[:], num_samples=1)

                for j in range(inputs_id.shape[0]):
                    if i != 0 and temp_sentence[j][-1] == eos[0]: continue
                    temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
                    model_train_CrossEntropy[j] = temp_loss + model_train_CrossEntropy[j]

                if i == 0:
                    for j in range(len(inputs_id)):    
                        temp_sentence[j].append(prev_input[j].item())
                    continue
                flag = 1
                
                for j in range(0, inputs_id.shape[0]):
                    if temp_sentence[j][-1] != eos[0]: 
                        flag = 0
                        temp_sentence[j].append(prev_input[j].item())

                if flag == 1: break
            
            decode_sentence = []
            for x in temp_sentence:
                decode_sentence.append(tokenizer.decode(x[:], skip_special_tokens=True).replace('<|endoftext|>', ''))
            # decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]


            reward = []
            for s in decode_sentence : 
                tmp_1, tmp_2, gen = replace_sentence(s)
                if gen == False:
                    reward.append(0)
                else:
                    r = make_reward(model_inter, tokenizer_inter, [tmp_1, tmp_2], analyzer,  device)
                    reward.append(r)

                    ######### Log ##############
                    if reward != 0:
                        f.write(f"{tmp_1}\n")
                        f.write(f"{tmp_2}\n")
                        f.write("="*10 + "\n")
                    count += 1
                    ############################

            reward = np.array(reward)
            reward = (reward - np.mean(reward)) / len(reward)
            for j in range(inputs_id.shape[0]) :
                loss = loss + model_train_CrossEntropy[j] * reward[j] 

            

            #### calculate loss
            loss.backward()
            batch_reward += (np.sum(reward) / 4)
            batch_loss += loss.item() / 4

            if (batch % 4 == 0 and batch != 0) or (batch + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"loss": batch_loss, "reward": batch_reward})
                pbar.set_postfix({'reward': batch_reward, 'loss': batch_loss})
                batch_loss = 0
                batch_reward = 0
        
            batch += 1
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_train.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"training_output/{args.save}/models/{epoch}.pt")

if __name__ == "__main__" :
    args = get_train_args()
    main(args)

