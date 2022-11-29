This project is modified from https://github.com/jacksukk/Chatbot-Project

## Get started
### 1. clone the repository
```
git clone https://github.com/FarnHua/chatbot_project.git
```

### 1. Proprocess 
```
python preprocess.py
```
Execute (or modify some part) of the above python file to proprocess keywords to json file required in training. 
(.txt -> .json) the format is showing below. 
```
{ topic : {'label1' : [keywords]
            'label2' : [keywords]
          }
}
```
Keywords now using words from https://github.com/uclanlp/gn_glove/tree/master/wordlist

### 2. traininig
```
python train_c.py
```
will currently execute training (will add arguments after). 
