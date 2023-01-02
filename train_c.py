import os
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import join
import math
from argparse import ArgumentParser
from DailyData import Daily

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

######################## Key words ##############################
keywords = {
}

with open('keywords/keys.json') as fp :
    keywords = json.load(fp)

######################### load keywords #########################

def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
       # print(values.shape)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    return logits
temperature = 1 #2.2
top_k = 50        #50
top_p = 0.95
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
def train(model_train, inputs_id, mask, tokenizer, ll, args, batch_size, analyzer):
    loss = 0
    inputs_id = inputs_id.to(device_0)
    
    eos = [tokenizer.encoder["<|endoftext|>"]]
    mask = mask.to(device_0)
    # prev_input, past = model_train(inputs_id, past=None, attention_mask=mask)
    model_train_out = model_train(inputs_id, past_key_values=None, attention_mask=mask)
    prev_input, past = model_train_out['logits'], model_train_out['past_key_values']
    inputs_id = inputs_id.to(device_1)
    mask = mask.to(device_1)

    # with torch.no_grad():
    #     prev_input, past_bot = model_2(inputs_id, past=None, attention_mask=mask)
    prev_input = torch.LongTensor([[eos] * inputs_id.shape[0]]).to(device_0)

    temp_sentence = [[] for i in range(inputs_id.shape[0])]
    model_train_CrossEntropy = [0 for i in range(inputs_id.shape[0])]
    # coherence_loss = [0 for i in range(inputs_id.shape[0])]
    append = torch.tensor([[1] for i in range(len(inputs_id))]).to(device_0)
    mask = torch.cat((mask, append), 1)

    for i in range(40):
        prev_input = prev_input.to(device_0)
        model_train_out = model_train(prev_input, past_key_values=past)
        logits, past = model_train_out['logits'], model_train_out['past_key_values']
        prev_input = prev_input.to(device_1)

        # with torch.no_grad():
        #     logits_bot, past_bot = model_2(prev_input, past=past_bot)
        mask = torch.cat((mask, append), 1)
        logits = logits.squeeze(0).squeeze(1)
        logits = logits / temperature

        logits = torch.softmax(logits, dim=-1)
        # with torch.no_grad():
        #     logits_bot = torch.softmax(logits_bot.squeeze(0).squeeze(1) / temperature, dim=-1)
        prev_input = torch.multinomial(logits[:], num_samples=1)



        ####################### calculate loss ############################
        for j in range(inputs_id.shape[0]):
            if i != 0 and temp_sentence[j][-1] == eos[0]: continue
            temp_loss = F.cross_entropy(logits[j].unsqueeze(0), prev_input.view(-1)[j].unsqueeze(0))
            # coherence_loss[j] += (logits_bot[j][prev_input[j][0].item()].item() - avg_prob) * temp_loss
            model_train_CrossEntropy[j] = temp_loss + model_train_CrossEntropy[j]

        ###################### append tokens to temp_sentences #############
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
    
    # print(temp_sentence)
    # assert(0)
    decode_temp_sentence = [tokenizer.decode(x).lower() for x in temp_sentence]
    # print(decode_temp_sentence)
    # for sent in decode_temp_sentence :
    #     sent = sent.lower()

    # print('Decode Sentence : ', decode_temp_sentence

    sent_1, sent_2 = [], []
    for x in decode_temp_sentence :
        tmp_1, tmp_2 = replace_sentence(x)
        sent_1.append(tmp_1)
        sent_2.append(tmp_2)    
    # print('sent1 : ', sent_1)
    # print('sent2 : ', sent_2)
    
    score_1 = []
    score_2 = []
    for sentence in sent_1 :
        vs_1 = analyzer.polarity_scores(sentence)
        score_1.append(vs_1['compound'])
    
    for sentence in sent_2 :
        vs_2 = analyzer.polarity_scores(sentence)
        score_2.append(vs_2['compound'])
    
    reward = []

    for j in range(inputs_id.shape[0]) :
        reward.append(abs(score_1[j] - score_2[j]))
        if abs(score_1[j] - score_2[j]) :
            print("Here is a non-zero score !")
    reward = np.array(reward)

    loss = 0
    for j in range(inputs_id.shape[0]) :
        loss = loss + model_train_CrossEntropy[j] * reward[j]
    
    # print(type(loss))
    # print('find ', key_count, ' key words. ')
    
    return loss, np.sum(reward)

   
def main():
    parser = ArgumentParser()
    parser.add_argument("--emotion", type=str, default="angry")
    parser.add_argument("--writer", type=str, default="")
    parser.add_argument("--save", type=str, default="model/save/")
    parser.add_argument("--model", type=str, default='microsoft/DialoGPT-small')
    parser.add_argument("--ra", type=float, default=3)
    parser.add_argument("--topic", type=str, default='gender')
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    # parser.add_argument("--inter", type=str, default="gpt", nargs='+', required=True)
    args = parser.parse_args()

    # os.makedirs('model/' + args.model, exist_ok=True)
    

    np.random.seed(100)
    torch.random.manual_seed(100)
    torch.cuda.manual_seed(100)
    model_train = GPT2LMHeadModel.from_pretrained(args.model)
    if args.pretrain :
        model_train.load_state_dict(torch.load(args.pretrain))
    # model_2 = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # dataset = load_dataset(
    # "daily_dialog",
    #     revision="main"  # tag name, or branch name, or commit hash
    # )
    # print(dataset)
    # assert(0)
    # if 'gpt' in args.inter:
    #     model_bot = GPT2LMHeadModel.from_pretrained('models/medium/')
    #     model_bot.to(device_1)
    #     model_bot.eval()
    #
    # if 'google' in args.inter:
    #     from main1 import chatbot
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) 
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #     jack = chatbot.Chatbot()
    #     jack.main(['--test', 'daemon', '--rootDir', 'deepqa', '--maxLength', '20'])
    # if 'retrieve' in args.inter:
    #     with torch.no_grad():
    #         from retrieval_model.retrieval_chatbot import Retrievalchatbot
    #         ret_model = Retrievalchatbot()
    writer = SummaryWriter('runs/'+args.writer+'/')
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

    model_train.to(device_0)
    # model_2.to(device_1)
    # model_2.eval()
    batch_size = 1
    
    # post = post_set('data/train_raw.tsv', tokenizer)
    post = Daily('data/daily_train_key.json', tokenizer, args)
    train_dataloader = DataLoader(post, batch_size=batch_size, shuffle=True, num_workers=1)
    batch = 0
    temp_score = 0
    loss = 0
   
    test_score = 0
    analyzer = SentimentIntensityAnalyzer()
    for global_step in range(1):
        model_train.train()
        # tmp_idx = 0
        for inputs_id, mask, ll in tqdm(train_dataloader):
            # if tmp_idx <= 20 :
            #     print(inputs_id)
            # tmp_idx += 1
            batch += 1
            batch_loss, score = train(model_train, inputs_id, mask, tokenizer, ll, args, batch_size, analyzer)
            loss += batch_loss

            # test_score += avg_prob
            temp_score += score
            # print('loss : ', loss)
            if batch % 4 == 0:
                loss.backward()
                # print("success backward")
                optimizer.step()
                # print("success step")
                writer.add_scalar('loss', loss, batch)
                optimizer.zero_grad()  
                loss = 0
            if batch % 20 == 0:
                # writer.add_scalar('reward', temp_score/batch_size/20, batch)
                # writer.add_scalar('test_reward', test_score/20, batch)
                # print("Reward:%.2f,    test:%.6f   "%(temp_score/batch_size/20/3, test_score/20))
                # test_score = 0
                temp_score = 0
            # if batch % 2500 == 0:
            #     torch.save(
            #         {k: (v.cpu() if v is not None else None)  # save to cpu tensors
            #             for k, v in model_train.state_dict().items()},
            #         join(f'model/{args.save}/',
            #                 f'{args.save}-{batch}.pkl'))

if __name__ == "__main__":
    main()
