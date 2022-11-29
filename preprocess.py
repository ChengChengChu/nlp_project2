import json

path = 'keywords/'

to_write = {
    'gender': {'men' : [], 'women' : []}
}

with open(path + 'men.txt') as fp :
    for line in fp.read().splitlines() :
        to_write['gender']['men'].append(line)

with open(path + 'women.txt') as fp :
    for line in fp.read().splitlines() :
        to_write['gender']['women'].append(line)

with open(path + 'keys.json', 'w') as fp :
    json.dump(to_write, fp)