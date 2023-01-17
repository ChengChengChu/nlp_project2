from argparse import ArgumentParser
import torch
import numpy as np
import random
import os

model_map = {
    'gpt': 'gpt2',
    'diologpt': 'microsoft/DialoGPT-small'
}
mens = []
womens = []
men_keys_to_idx = {}
women_keys_to_idx = {}


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_finetune(args):
    DIR = './pretrain_output'
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(os.path.join(DIR, args.save), exist_ok=True)
    os.makedirs(os.path.join(DIR, args.save, "models"), exist_ok=True)
    
def set_train(args) :
    idx = 0
    with open('keywords/men.txt') as fp :
        idx = 0
        for line in fp.read().splitlines() :
            mens.append(line.lower())
            men_keys_to_idx[line.lower()] = idx
            idx += 1
    
    with open('keywords/women.txt') as fp : 
        idx = 0
        for line in fp.read().splitlines() :
            womens.append(line.lower())
            women_keys_to_idx[line.lower()] = idx
            idx += 1
    
    DIR = './training_output'
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(os.path.join(DIR, args.save), exist_ok=True)
    os.makedirs(os.path.join(DIR, args.save, "models"), exist_ok=True)

def get_finetune_args():
    
    parser  = ArgumentParser()
    parser.add_argument("--save", type=str, default="gpt_pretrain")
    parser.add_argument("--model", type=str, default='gpt')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epoch",type=int,default=2)
    parser.add_argument("--mode",type=str,default='train')
    parser.add_argument("--ckpt",type=str,default=None)
    parser.add_argument("--prompt",type=str,default="where is john ? i can't find him anywhere .")
    parser.add_argument("--lr",type=float,default=2e-5)
    args = parser.parse_args()
    return args

def get_train_args():
    
    parser  = ArgumentParser()
    parser.add_argument("--save", type=str, default="gpt_train")
    parser.add_argument("--model", type=str, default='gpt')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epoch",type=int,default=2)
    parser.add_argument("--ckpt",type=str,default=None)
    parser.add_argument("--lr",type=float,default=2e-5)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()
    return args


def generate(
    model,
    tokenizer,
    prompt, 
    device,
    max_length=40, #maximum number of words
    top_k=50,
    temperature=.9,
):  
    model.eval()
    eos = [tokenizer.encoder["<|endoftext|>"]]
    inputs_id = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    # inputs_id.to(device)
    model = model.to(device)
    model_out = model(inputs_id)
    prev_input, past = model_out['logits'], model_out['past_key_values']

    prev_input = torch.LongTensor([eos[0]]).to(device)
    # prev_input = torch.LongTensor(tokenizer.encode(['<|endoftext|>'])).to(device)
    temp_sentence = []

    with torch.no_grad():
        for i in range(max_length):
            prev_input = prev_input.to(device)
            
            model_train_out = model(prev_input, past_key_values=past)
            logits, past = model_train_out['logits'], model_train_out['past_key_values']
            
            logits = logits.squeeze(0) # shape:(50256,)
            ## top_k
            filter_value = -float('inf')
            values, _ = torch.topk(logits, top_k)
            min_values = values[-1].repeat(logits.shape[-1])
            logits = torch.where(logits < min_values, 
                        torch.ones_like(logits, dtype=logits.dtype) * filter_value, 
                        logits)

            logits = logits / temperature
            logits = torch.softmax(logits, dim=-1)
            #####
            prev_input = torch.multinomial(logits[:], num_samples=1)

            if i == 0:              
                temp_sentence.append(prev_input[0].item())
                continue
            flag = 1
            if temp_sentence[-1] != eos[0]: 
                flag = 0
                temp_sentence.append(prev_input[0].item())

            if flag == 1: break
    
    # decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]
    decode_temp_sentence = tokenizer.decode(temp_sentence).replace("<|endoftext|>", "")
        
                
    return decode_temp_sentence
    
def replace_sentence(sens) :

    ''' This function returns two sentences correspond to the given sentence
        str --> str, str

        e.g. 
        He is my father  --> He is my father, She is my mother
    '''
    ret_1 = " "
    ret_2 = " "

    key_word_idx = []

    sens = sens.replace('\n', '') + '\n'

    sens_without_period = []
    
    sens = [x.lower() for x in sens.split()]

    period = [',', '.', '!', '?', '<', '>', '~', '{', '}', '[', ']', "'", '"', ':']
    for s in sens:
        s_ = s
        for p in period:
            s_ = s_.replace(p, '')
        sens_without_period.append(s_)

    assert(len(sens_without_period) == len(sens))

    # find key word list 
    for i in range(len(sens_without_period)) : 
        # print(sens_without_period[i] + '|')
        if sens_without_period[i] in mens or sens_without_period[i] in womens :
            # print("PASS")
            key_word_idx.append(i)
    
    ret_1 = sens[:]
    ret_2 = sens[:]
    gen = False
    for i in key_word_idx :
        tmp = sens_without_period[i]
        if tmp in womens :
            ret_1[i] = ret_1[i].replace(tmp, mens[women_keys_to_idx[tmp]])
            gen = True
        
        if tmp in mens :
            ret_2[i] = ret_2[i].replace(tmp, womens[men_keys_to_idx[tmp]])
            gen = True
    
    return " ".join(ret_1), " ".join(ret_2), gen
