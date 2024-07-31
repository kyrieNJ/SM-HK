# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
import os
import pickle
from queue import Queue

import spacy

from vocab import Vocab
import numpy as np
from torch.utils.data import Dataset
import torch
import _pickle as cPickle
from transformers import BertTokenizer
from tqdm import tqdm
import json

tokenizer = BertTokenizer.from_pretrained("./mlm/bert_finetune_sm_final")
nlp = spacy.load('zh_core_web_sm')

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

def text_to_bert_sequence(text, max_len, padding="post", truncating="post"):
    text = tokenizer.tokenize(text)
    text = ["[CLS]"] + text + ["[SEP]"]
    sequence = tokenizer.convert_tokens_to_ids(text)
    return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc

    return x

def load_KGword_vec(path):
    fin = pickle.load(open(path, 'rb'))
    kgword_vec = {}
    for line in range(1,len(fin)+1):
        kgword_vec[line] = np.asarray(fin[line], dtype='float32')
    #11，768
    return kgword_vec

def build_SYembedding_matrix_BERT():
    KGembedding_matrix_file_name = 'SYembedding_matrix_BERT.pkl'
    if os.path.exists(KGembedding_matrix_file_name):
        print('loading embedding_matrix:', KGembedding_matrix_file_name)
        SYembedding_matrix = pickle.load(open(KGembedding_matrix_file_name, 'rb'))
    else:
        SYembedding_matrix = np.zeros((12, 768))
        SYembedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(768), 1/np.sqrt(768), (1, 768))
        SYembedding_matrix_file_name = 'SYembedding_dict_BERT.pkl'
        kgword_vec = load_KGword_vec(SYembedding_matrix_file_name)
        print('building embedding_matrix:', KGembedding_matrix_file_name)
        for i in range(1,len(kgword_vec)+1):
            vec = kgword_vec[i]
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                SYembedding_matrix[i] = vec
        # pickle.dump序列化对象，将对象obj保存到文件file中去。
        pickle.dump(SYembedding_matrix, open(KGembedding_matrix_file_name, 'wb'))
    return SYembedding_matrix


class PDUDDataset(Dataset):
    def __init__(self, fname,args):

        fin = open(fname + '_u_twi_symlist'+'_'+str(args.symp_v)+'sy'+str(args.sim_v)+'si'+'_'+str(args.topk)+'.raw', 'rb')
        u_twi_symlist = pickle.load(fin)
        fin.close()
        fin = open(fname + '_u_graph'+'_'+str(args.symp_v)+'sy'+str(args.sim_v)+'si'+'_'+str(args.topk)+'.graph', 'rb')
        u_graph = pickle.load(fin)
        fin.close()


        self.SYembedding_matrix = build_SYembedding_matrix_BERT()

        user_data_dict = parse_json_all(fname)


        user_count=0
        all_data = []
        maxtextlen=0
        maxtwilen=0
        for iuser in tqdm(user_data_dict):
            twi_text_bertlist=[]

            twi_count=0
            for itw in iuser['tweets']:
                twi_text = itw['tweet_content'].strip()

                bert_text_sequence = text_to_bert_sequence(twi_text, args.max_len)
                twi_text_bertlist.append(bert_text_sequence)
                if maxtextlen<len(bert_text_sequence):
                    maxtextlen=len(bert_text_sequence)
                    print(maxtextlen)

                twi_count+=1

            sym_list = u_twi_symlist[user_count]
            sym_padding = [0] * (args.sy_max_len - len(sym_list))
            sym_add = np.array(sym_list + sym_padding)

            ugraph = np.array(u_graph[user_count])
            len_u_gra = len(ugraph)
            if maxtwilen < len_u_gra:
                maxtwilen = len_u_gra
                print("len_u_gra",len_u_gra)

            ulable=iuser['risk_label']
            smlable=iuser['SMlabel']

            list_pad = [0] * (args.max_len)
            list_pad = np.array(list_pad)

            for i in range(0,args.twi_max_len - len(twi_text_bertlist)):

                twi_text_bertlist.append(list_pad)


            data = {
                'twi_text_bert_sequence_list':np.array(twi_text_bertlist),
                'u_sym_list':sym_add,
                'u_twi_len':np.array(twi_count),
                'u_graph':np.pad(ugraph, (
                (0, args.twi_max_len - len_u_gra), (0, args.twi_max_len - len_u_gra)), 'constant'),
                'risk_label': int(ulable),
                'sm_lable': int(smlable),
            }
            all_data.append(data)
            user_count+=1
        print('maxtwilen:', maxtwilen)
        print('maxtextlen:', maxtextlen)
        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)