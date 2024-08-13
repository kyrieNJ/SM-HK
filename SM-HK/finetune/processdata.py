import numpy as np
import spacy
import pickle
import _pickle as cPickle
import os
import json
from tqdm import tqdm
if os.path.getsize('../senticnet_zh/senticdict.pkl') > 0:
    with open('../senticnet_zh/senticdict.pkl', 'rb') as f:
        sentic = pickle.load(f)
nlp = spacy.load('zh_core_web_sm')

def processall_sm(filename):
    with open(filename, 'r',encoding='utf-8') as f:
        traindata = json.load(f)

    listpro=[]
    for iuser in tqdm(traindata):
        user_text=''
        sta_list=[]
        for ista in iuser['tweets']:
            itext = ista['tweet_content']
            if itext not in sta_list:
                sta_list.append(itext)
                document2 = nlp(itext)
                user_text=itext
                # Sentence Sentic
                sentlist = []

                for j in document2:
                    if j.pos_ == 'NOUN' or j.pos_ == 'VERB':
                        if j.text not in sentic.keys():
                            continue
                        tmp_word_poa=sentic[j.text][1]
                        if tmp_word_poa == 'positive':
                            continue
                        tmp_word_score = float(sentic[j.text][2])
                        tmpserve=''
                        if tmp_word_score > -0.25 and tmp_word_score <= 0:
                            tmpserve = '轻微的'
                        elif tmp_word_score > -0.55 and tmp_word_score <= -0.25:
                            tmpserve = '中等的'
                        elif tmp_word_score > -1 and tmp_word_score <= -0.55:
                            tmpserve = '严重的'

                        aux_sent = '词“' + j.text + '”可能具有'+tmpserve+'消极情感。'

                        if aux_sent not in sentlist:
                            sentlist.append(aux_sent)
                tmp_text=''
                for itemp in sentlist:
                    tmp_text=tmp_text+itemp

                user_text+='  '+tmp_text
            if user_text not in listpro:
                listpro.append(user_text)
    return listpro

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

def processall():
    listproall=[]
    dataset_type = ['RP_train.json']


    fout = open('./pretraintext/finetune_alltext_sm_final.tsv', 'w', encoding = 'utf-8')
    for idt in dataset_type:
        print(idt)
        pathname='../datasets/riskPredict/'+idt
        li_i=processall_sm(pathname)
        listproall.extend(li_i)
    for i in listproall:
        fout.write(i+'\n')
    fout.close()

processall()