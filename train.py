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
from undecorated import undecorated
from types import MethodType
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
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    if name == "gen":
        if args.ckpt != None: 
            print("Using model with finetuning !!!")
            model.load_state_dict(torch.load(args.ckpt))
        else : 
            print("Training with base model without finetuning !!!")
    
    return model, tokenizer


def make_reward(sen1, sen2, analyzer, model_inter, tokenizer_inter, device):

    model_inter.eval()
    sen1 = torch.tensor(tokenizer_inter.encode(sen1)).unsqueeze(0).to(device)
    sen2 = torch.tensor(tokenizer_inter.encode(sen2)).unsqueeze(0).to(device)
    mask1 = torch.ones(sen1.shape).to(device)
    mask2 = torch.ones(sen2.shape).to(device)
    
    generate1 = model_inter.generate(
                                sen1,
                                attention_mask=mask1,
                                do_sample=True,   
                                top_k=40, 
                                max_length=45,
                                # top_p=0.95, 
                                num_return_sequences=1,
                                pad_token_id=50256
                            )

    generate2 = model_inter.generate(
                                sen2,
                                attention_mask=mask2,
                                do_sample=True,
                                top_k=40,
                                max_length=45,
                                num_return_sequences=1,
                                pad_token_id=50256
                            )
    
    vs_1 = analyzer.polarity_scores(tokenizer_inter.decode(generate1[0], skip_special_tokens=True))
    vs_2 = analyzer.polarity_scores(tokenizer_inter.decode(generate2[0], skip_special_tokens=True))
    return abs(vs_1['compound'] - vs_2['compound'])

def main(args) :
    set_wandb(args)
    set_train(args)
    analyzer = SentimentIntensityAnalyzer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model_train, tokenizer = set_model(args, "gen")
    model_train = model_train.to(device)

    generate_with_grad = undecorated(model_train.generate)
    model_train.generate_with_grad = MethodType(generate_with_grad, model_train)

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
    f = open(f"training_output/{args.save}/log.txt", "w")
    count = 0
    for epoch in range(args.epoch):
        batch = 0
        loss = 0
        pbar = tqdm(train_dataloader)
        batch_loss = 0
        batch_reward = 0

        # inputs_id : B x Seq
        for inputs_id, mask, length in pbar:
            # import pdb
            # pdb.set_trace()
            sample_outputs = model_train.generate_with_grad(
                                inputs_id[:,0].unsqueeze(0).to(device), 
                                #bos_token_id=random.randint(1,30000),
                                attention_mask=torch.LongTensor([1]).unsqueeze(0).to(device), 
                                do_sample=True,   
                                top_k=40, 
                                max_length = 40,
                                # top_p=0.95, 
                                num_return_sequences=1, 
                                # return_dict=True
                                output_scores=True,
                                return_dict_in_generate=True,
                                pad_token_id=50256
                                )

            cross_entropy = 0
            idx = 0
            for x in sample_outputs['scores'] :
                p = torch.softmax(x, dim=1)
                cross_entropy = cross_entropy + F.cross_entropy(p.squeeze(0), sample_outputs['sequences'][0][idx])
                idx += 1
            decode_sentence = tokenizer.decode(sample_outputs["sequences"][0], skip_special_tokens=True)
            
            tmp_1, tmp_2, gen = replace_sentence(decode_sentence)

            if gen == False:  
                continue

            reward = make_reward(tmp_1, tmp_2, analyzer, model_inter, tokenizer_inter, device)
            
            ######### Log ##############
            if reward != 0:
                f.write(f"{tmp_1}\n")
                f.write(f"{tmp_2}\n")
                f.write("="*10 + "\n")
            count += 1
            ############################

            loss = reward * cross_entropy
            batch_reward += (reward / 32)
            loss.backward()
            batch_loss += loss.item() / 32

            if (batch % 32 == 0 and batch != 0) or (batch + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"loss": batch_loss, "reward": batch_reward})
                pbar.set_postfix({'reward': batch_reward, 'loss': batch_loss})
                batch_loss = 0
                batch_reward = 0
            batch += 1
        
        count /= len(train_dataloader)
        f.write(f"gender rate: {count * 100}%\n")
        f.close()


if __name__ == "__main__" :
    args = get_train_args()
    main(args)

