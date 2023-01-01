import json

path = 'data/daily_train.json'
key_path = 'keywords/keys.json'

keys = []

data = {}
with open(path) as fp :
    data =json.load(fp)

#### get keywords 
tmp = {}
with open(key_path) as fp :
    tmp = json.load(fp)

for k in tmp['gender']['men'] :
    keys.append(k)

for k in tmp['gender']['women'] :
    keys.append(k)

count = 0
data_keys = {'dialog' : []}

for sens in data['dialog'] :
    # print(sens[0])
    
    for s in sens:
        tmp_list = s.lower()
        for k in keys :
            if k in tmp_list.split() :
                if [" ".join([i for i in tmp_list.split()])] not in data_keys['dialog'] :
                    data_keys['dialog'].append([" ".join([i for i in tmp_list.split()])])

print(len(data_keys['dialog']))

with open('data/daily_train_key.json', 'w') as fp :
    json.dump(data_keys, fp)
    




