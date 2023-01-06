# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# test = ['Chihuahua is so cute', 'Fuck all of you']

# ### neg : negative, pos : positive
# analyzer = SentimentIntensityAnalyzer()
# for sentence in test:
#     vs = analyzer.polarity_scores(sentence)
#     # print(vs['neg'], vs['pos'])
#     print(vs)
#     # print("{:-<65} {}".format(sentence, str(vs)))

from utils import *

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
s = "certainty, sir."
x, y, gen = replace_sentence(s)

print(x)
print(y)
print(gen)

