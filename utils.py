from argparse import ArgumentParser
import torch
import numpy as np
import random
import os

model_map = {
    'gpt': 'gpt2',
    'diologpt': 'microsoft/DialoGPT-small'
}

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