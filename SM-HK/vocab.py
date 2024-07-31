import pickle
from tqdm import tqdm
from collections import Counter
from models.utils import (
        parse_term,
        preprocess_pretrain,
        preprocess_finetune,
        )
import os
import jieba
import json
import spacy
nlp = spacy.load('zh_core_web_sm')

class Vocab(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]
        
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        # self.itostr = {i: tok for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def getdata(train_path,dev_path,test_path,config):
    lowercase = config['lowercase'] if 'lowercase' in config else False
    remove_list = config['remove_list'] if 'remove_list' in config else []

    train_data = parse_term(train_path,
                            lowercase=lowercase, remove_list=remove_list)
    dev_data = parse_term(dev_path,
                          lowercase=lowercase,
                          remove_list=remove_list) if dev_path else None
    test_data = parse_term(test_path,
                           lowercase=lowercase,
                           remove_list=remove_list) if test_path else None
    return train_data,dev_data,test_data

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

if __name__ == '__main__':
    djsonfile = './datasets/riskPredict/new/old_WU3D_train.json'
    tmpdict = parse_json_all(djsonfile)
    wordlist=[]
    for idct in tqdm(tmpdict):
        # tmp_text+=' '+idct["label"]
        # tmp_text+=' '+idct["nickname"]
        # tmp_text+=' '+idct["profile"]
        for ire in idct['tweets']:
            tmp_text=ire['tweet_content']
            if tmp_text=='无' or tmp_text=='' or tmp_text=='无。':
                continue
            document2 = nlp(tmp_text)
            for idoc in document2:
                wordlist.append(idoc.text)


    counter={}
    for i in range(len(wordlist)):
        if wordlist[i] not in counter.keys():
            counter[wordlist[i]]=1
        else:
            counter[wordlist[i]]+=1
    counter['<pad>']=1
    counter['<unk>']=1
    token=Vocab(counter)
    fout='./all_vocab_tok.vocab'
    token.save_vocab(fout)
