from argparse import ArgumentParser
import torch
import numpy as np
import random
import os
from decoding import *

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
    
def set_train() :
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

def get_finetune_args():
    
    parser  = ArgumentParser()

    parser.add_argument(
        "--save", 
        type=str, 
        default="gpt_pretrain")

    parser.add_argument(
        "--model", 
        type=str, 
        default='gpt')

    parser.add_argument(
        "--batch", 
        type=int, 
        default=1)

    parser.add_argument(
        "--seed", 
        type=int, 
        default=100)

    parser.add_argument(
        "--epoch",
        type=int,
        default=2
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='train'
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None
    )

    args = parser.parse_args()
    return args

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=100
    )

    parser.add_argument(
        "--emotion",
        type=str,
        default=None
    )

    parser.add_argument(
        "--sw",
        type=str,
        default=None
    )

    parser.add_argument(
        "--writer", 
        type=str, 
        default="")

    parser.add_argument(
        "--save", 
        type=str, 
        help="save path",
        default="model/save/")
        
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/blenderbot-400M-distill")

    parser.add_argument(
        "--ra", 
        type=float, 
        default=3)

    parser.add_argument(
        "--inter", 
        type=str, 
        default="gpt", 
        nargs='+', 
        required=True)

    parser.add_argument(
        "--n_tokens", 
        type=int, 
        default=10)


    parser.add_argument(
        "--mode", 
        type=str, 
        help="length / emotion / sw",
        default='length')

    parser.add_argument(
        "--initial", 
        type=str, 
        default='vocab')

    parser.add_argument(
        "--top_k", 
        type=int, 
        default=0)

    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.0)

    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0)

    args = parser.parse_args()

    return args


def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_length=40, #maximum number of words
    top_p=0.8,
    top_k=0.95,
    temperature=1.,
):
    
    model.eval()

    inputs_id = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    inputs_id.to(device)
    model_out = model(inputs_id)
    prev_input, past = model_out['logits'], model_out['past_key_values']

    prev_input = torch.LongTensor(['<|endoftext|>'])
    temp_sentence = []

    with torch.no_grad():
        for i in range(max_length):
            prev_input = prev_input.to(device)
            model_train_out = model(prev_input, past_key_values=past)
            logits, past = model_train_out['logits'], model_train_out['past_key_values']

            logits = logits.squeeze(0).squeeze(1)
            logits = original(logits)
            prev_input = torch.multinomial(logits[:], num_samples=1)

            if i == 0:              
                temp_sentence.append(prev_input[0].item())
                continue
            flag = 1
            if temp_sentence[-1] != '<|endoftext|>': 
                flag = 0
                temp_sentence.append(prev_input[0].item())

            if flag == 1: break
    
    decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]
        
                
    return decode_temp_sentence
def replace_sentence(sens) :

    ''' This function returns two sentences correspond to the given sentence
        str --> str, str

        e.g. 
        He is my father  --> He is my father, She is my mother
    '''
    # print("PASS\n\n")
    ret_1 = " "
    ret_2 = " "

    key_word_idx = []

    sens_without_period = [x.lower() for x in sens.split('<|endoftext|>')[:-1]][0]
    sens = [x.lower() for x in sens.split('<|endoftext|>')[:-1]][0]

    period = [',', '.', '!', '?', '<', '>', '~', '{', '}', '[', ']', "'", '"']
    for p in period : 
        sens_without_period = sens_without_period.replace(p, '')
    
    sens_without_period = sens_without_period.replace('  ', ' ')
    sens_without_period = sens_without_period.split()

    # find key word list 
    for i in range(len(sens_without_period)) : 
        if sens_without_period[i] in mens or sens_without_period[i] in womens :
            key_word_idx.append(i)
    
    ret_1 = sens.split()
    ret_2 = sens.split()

    for i in key_word_idx :
        tmp = sens_without_period[i]

        if tmp in womens :
            ret_1[i] = ret_1[i].replace(tmp, mens[women_keys_to_idx[tmp]])
        
        if tmp in mens :
            ret_2[i] = ret_2[i].replace(tmp, womens[men_keys_to_idx[tmp]])
    
    return " ".join(ret_1), " ".join(ret_2)
