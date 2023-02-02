import torch
from torch import nn
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer




def original(logits, temperature = 1.0):
  logits = logits / temperature
  logits = torch.softmax(logits, dim=-1)

  return logits

def top_k_top_p_filtering(logits, top_k = 0, top_p = 0.0, temperature = 1.0):
  # logits = torch.softmax(logits, dim=-1)
  filter_value = -float('inf')

  if top_k > 0:
    values, _ = torch.topk(logits, top_k)
       # print(values.shape)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    logits = torch.where(logits < min_values, 
                torch.ones_like(logits, dtype=logits.dtype) * filter_value, 
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

  logits = logits / temperature
  logits = torch.softmax(logits, dim=-1)

  return logits

class bot(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
        self.lm.to(self.device)
        self.lm.eval()
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature

    def make_response(self, first_input):
        with torch.no_grad():
            sentences = []
            for i in range(len(first_input)):
                sentences.append((list(first_input[i])))
            m = []
            for i in range(len(sentences)):
                temp_m = [1 for x in range(len(sentences[i]))]
                m.append(temp_m[:])
            eos = [self.tokenizer.encoder["<|endoftext|>"]]

            # prepare original input to model
            prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in sentences], value=0)).to(self.device)
            m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0)).to(self.device)

            position_ids = m.long().cumsum(-1) - 1 #+ prev_input.shape[1]
            position_ids.masked_fill_(m == 0, 1)

            outputs = self.lm(prev_input, past_key_values=None, attention_mask=m, position_ids=position_ids)
            past = outputs['past_key_values']

            prev_input = torch.LongTensor([[eos] * len(sentences)]).squeeze(0).to(self.device)
            append = torch.tensor([[1] for i in range(len(sentences))]).to(self.device)
            m = torch.cat((m, append), 1)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            temp_sen = [[] for i in range(len(sentences))]
            
            

            for i in range(40):
                outputs = self.lm(prev_input, past_key_values=past, attention_mask=m, position_ids=position_ids)
                prev_input, past = outputs["logits"], outputs["past_key_values"]
                m = torch.cat((m, append), 1)
                position_ids = m.long().cumsum(-1) - 1
                position_ids.masked_fill_(m == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                prev_input = prev_input.squeeze(0).squeeze(1)
                prev_input = top_k_top_p_filtering(prev_input, top_k=self.top_k, temperature=self.temperature)
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
                a.append(self.tokenizer.decode(x[:], skip_special_tokens=True).replace('<|endoftext|>', ''))
            return a


