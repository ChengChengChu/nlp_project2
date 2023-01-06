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
import wandb

def set_wandb(args):
    wandb.init(
        project="bias_finetune",
        name=f"{args.save}",
        entity="chatbot_ntu"
    )
    wandb.config.update(args)


def set_model(args):

    if args.model in model_map:
        m = model_map[args.model]
    else:
        m = args.model
    
    model = GPT2LMHeadModel.from_pretrained(m)
    tokenizer = GPT2Tokenizer.from_pretrained(m, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

    if args.ckpt != None:
        model.load_state_dict(torch.load(args.ckpt))
    
    return model, tokenizer


def test_finetune(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.ckpt = "pretrain_output/gpt2-large/models/gpt2-large-4.pt"
    model, tokenizer = set_model(args)

    # prompt = args.prompt
    prompt = "Who"
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    model.to(device)
    model.eval()
    sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                attention_mask=torch.LongTensor([1]).unsqueeze(0).to(device), 
                                do_sample=True,   
                                top_k=40, 
                                max_length = 40,
                                # top_p=0.95, 
                                num_return_sequences=1, 
                                # return_dict=True
                                output_scores=True,
                                return_dict_in_generate=True
                                )

    cross_entropy = 0
    idx = 0
    for x in sample_outputs['scores'] :
        p = torch.softmax(x, dim=1)
        cross_entropy = cross_entropy + F.cross_entropy(p.squeeze(0), sample_outputs['sequences'][0][idx])
        idx += 1
        # break
    
    print(cross_entropy)
    # sample_outputs['sequences'] : batch * sequence_length (1 x 7)
    

    for i, sample_output in enumerate(sample_outputs['sequences']):
        print("{}: {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    # sentence = generate(model, tokenizer, prompt, device)

    # print(sentence)




def train(args) :

    ## set environment
    set_seed(args.seed)
    set_finetune(args)
    set_wandb(args)


    ## prepare model & optmizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    optimizer = Adam(optimizer_grouped_parameters, args.lr,
                     max_grad_norm=1.0)
    

    ## prepare dataset
    daily_data = Daily('data/daily_train_key.json', tokenizer, args)
    train_dataloader = DataLoader(daily_data, batch_size=args.batch, shuffle=True, num_workers=4)
    
    
    
    ## training
    model_train.train()
    
    for epoch in range(args.epoch):
        batch = 0
        batch_loss = 0
        pbar = tqdm(train_dataloader)
        for inputs_id, mask, length in pbar:

            inputs_id = inputs_id[:, :length].to(device)
            output = model_train(inputs_id, labels=inputs_id)
            loss = output['loss']
            loss = loss / 64
            batch_loss += loss.item()
            loss.backward()

            if batch % 64 == 0 or (batch + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({'epoch': epoch, 'loss': batch_loss})
                wandb.log({"loss": batch_loss})
                with open(f"./pretrain_output/{args.save}/loss.txt", 'a') as f:
                    f.write(f"[Epoch: {epoch} | step: {batch/64}]: {batch_loss}\n")
                batch_loss = 0
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