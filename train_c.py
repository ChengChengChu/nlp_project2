import numpy as np
from torch.utils.data import DataLoader
from DailyData import Daily
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm
import torch
import torch.nn.functional as F
from lsp_model.optim import *
from tqdm import tqdm
from utils import *
from decoding import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import wandb
import importlib

def set_wandb(args):
    wandb.init(
        project="bias_phase1",
        name=f"{args.save}",
        entity="chatbot_ntu"
    )
    wandb.config.update(args)


def set_model(args) :

    
    if args.model in model_map:
        m = model_map[args.model]
    else :
        m = args.model
    
    model = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m)

    
    if args.ckpt != None: 
        print("Using model with finetuning !!!")
        model.load_state_dict(torch.load(args.ckpt))
    else : 
        print("Training with base model without finetuning !!!")
    
    return model, tokenizer

def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty=1.2):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(len(prev_output_tokens)):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty
    return lprobs


def make_reward(a, analyzer):
    

        
    vs_1 = analyzer.polarity_scores(a[0])
    vs_2 = analyzer.polarity_scores(a[1])
    return abs(vs_1['compound'] - vs_2['compound'])


def main(args) :
    
    analyzer = SentimentIntensityAnalyzer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    set_train(args)
    set_wandb(args)

    model_train, tokenizer = set_model(args)
    model_train = model_train.to(device)

    model_co, tokenizer = set_model(args)
    model_co = model_co.to(device)
    

    # model_inter, tokenizer_inter = set_model(args, "inter")
    # model_inter = model_inter.to(device)

    inter = importlib.import_module(".module", f"bots.{args.inter}").bot
    model_inter = inter(args, device)

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
    model_co.eval()
    model_inter.eval()
    f = open(f"training_output/{args.save}/log.txt", "w")
    count = 0
    total = 0
    for epoch in range(args.epoch):
        batch = 0
        
        pbar = tqdm(train_dataloader)
        batch_loss = 0
        batch_reward = 0
        

        for inputs_id, mask, length in pbar:
            loss = 0
            prev_input = inputs_id[:,0].unsqueeze(1).to(device)
            m = mask[:,0].unsqueeze(1).to(device)
            eos = [tokenizer.encoder["<|endoftext|>"]]
            total += args.batch
            # prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in inputs_id], value=0)).to(device)
            # m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in mask], value=0)).to(device)

            # position_ids = m.long().cumsum(-1) - 1 #+ prev_input.shape[1]
            # position_ids.masked_fill_(m == 0, 1)
            
            append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)

            position_ids = position_ids[:, -1].unsqueeze(-1)

            temp_sentence = [[] for i in range(inputs_id.shape[0])]

            ### for coherence ###
            coherence_loss = [0 for i in range(inputs_id.shape[0])]
            test_reward = [1 for i in range(inputs_id.shape[0])]
            #####################

            model_train_CrossEntropy = [0 for i in range(inputs_id.shape[0])]

            for j in range(len(prev_input)):
                temp_sentence[j].extend(prev_input[j])
            past = None
            past_co = None
            
            for i in range(40): 
                model_train_out = model_train(prev_input, attention_mask=m, past_key_values=past, position_ids=position_ids)
                logits, past = model_train_out['logits'], model_train_out['past_key_values']

                with torch.no_grad():
                    model_co_out = model_co(prev_input, past_key_values=past_co, attention_mask=m, position_ids=position_ids)
                    logits_co, past_co = model_co_out['logits'], model_co_out['past_key_values']


                m = torch.cat((m, append), 1)
                position_ids = m.long().cumsum(-1) - 1
                position_ids.masked_fill_(m == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                logits = logits.squeeze(0).squeeze(1)
                logits = top_k_top_p_filtering(logits, top_k=args.top_k, temperature=2.2) 
                prev_input = torch.multinomial(logits[:], num_samples=1)

                with torch.no_grad():
                    logits_co = logits_co.squeeze(0).squeeze(1)
                    logits_co = top_k_top_p_filtering(logits_co, top_k=args.top_k, temperature=2.2)
                
                ########## coherence ##########
                probs = []
                for j in range(inputs_id.shape[0]):
                    if i != 0 and temp_sentence[j][-1] == eos[0]:
                        continue
                    prob = logits_co[j][prev_input[j][0].item()].item()
                    probs.append(prob)
                    test_reward[j] *= prob
                if len(probs) == 0:
                    avg_prob = 0
                else:
                    avg_prob = sum(probs) / len(probs)
                ###############################

                for j in range(inputs_id.shape[0]):
                    if i != 0 and temp_sentence[j][-1] == eos[0]: continue
                    temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
                    coherence_loss[j] += (logits_co[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
                    model_train_CrossEntropy[j] += temp_loss

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
                decode_sentence.append(tokenizer.decode(x, skip_special_tokens=True).replace('<|endoftext|>', ''))
            # decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]
            
            # import pdb
            # pdb.set_trace()

            reward = []
            for s in decode_sentence : 
                tmp_1, tmp_2, gen = replace_sentence(s)
                tmp_1_encode = tokenizer.encode(tmp_1)
                tmp_2_encode = tokenizer.encode(tmp_2)
                if gen == False:
                    reward.append(0)
                else:
                    # r1, r2, r = make_reward(args, model_inter, tokenizer_inter, [tmp_1_encode, tmp_2_encode], analyzer,  device)
                    responses = model_inter.make_response([tmp_1_encode, tmp_2_encode])
                    r = make_reward(responses, analyzer)
                    reward.append(r)

                    ######### Log ##############
                    if reward != 0:
                        f.write(f"{tmp_1}\n{responses[0]}\n{tmp_2}\n{responses[1]}\n")
                        f.write("="*10 + "\n")
                        count += 1
                    ############################
            # import pdb
            # pdb.set_trace()
            
            reward = np.array(reward)
            avg_reward = np.mean(reward)
            reward = (reward - avg_reward)
            
            for j in range(inputs_id.shape[0]) :
                loss += model_train_CrossEntropy[j] * reward[j] 
                loss += coherence_loss[j] * args.ra

            

            #### calculate loss
            batch_reward += (avg_reward / 4)
            batch_loss += loss.item() / 4
            loss.backward()
            

            if (batch % 4 == 0 and batch != 0) or (batch + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"loss": batch_loss, "reward": batch_reward, "hit": count / total})
                pbar.set_postfix({'reward': batch_reward, 'loss': batch_loss, 'hit':count / total})
                batch_loss = 0
                batch_reward = 0
                
        
            batch += 1
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_train.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"training_output/{args.save}/models/{epoch}.pt")
    

if __name__ == "__main__" :
    torch.cuda.empty_cache()
    args = get_train_args()
    main(args)

