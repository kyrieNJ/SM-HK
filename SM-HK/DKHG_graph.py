# -*- coding: utf-8 -*-
import random

import numpy as np
import spacy
import pickle
import time
import jieba
import os
from vocab import  Vocab
import json
from tqdm import tqdm
nlp = spacy.load('zh_core_web_sm')
# nlp = spacy.load('zh_core_web_trf')
import torch
from transformers import BertModel, BertConfig
import Bert_representation
dep_list = []
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#BERT
bert = BertModel.from_pretrained("./bert-chinese")
# bert = BertModel.from_pretrained("./mlm/bert-de_finetune")
device = torch.device(device=1)
# device = torch.device('cpu')
Bert_encoder = bert  # Bert模型
Bert_encoder.to(device)
Bert_encoder.eval()

if os.path.getsize('./senticnet_zh/senticdict.pkl') > 0:
    with open('./senticnet_zh/senticdict.pkl', 'rb') as f:
        sentic = pickle.load(f)

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict


def processHG(filename,ds_rep_dict,symp_v,sim_v,topk,senti_rep_dict,senti_idx_dict,idx_count):

    tmpdict = parse_json_all(filename)

    u_twi_symlist={}
    u_graph = {}
    u_gcngraph={}
    foututs = open(filename +'_u_twi_symlist'+'_'+str(symp_v)+'sy'+str(sim_v)+'si'+'_'+str(topk)+'.raw', 'wb')
    foutug = open(filename+'_u_graph'+'_'+str(symp_v)+'sy'+str(sim_v)+'si'+'_'+str(topk)+ '.graph', 'wb')

    foutgg = open(filename+'_u_gcngraph'+ '.graph', 'wb')

    count=0
    for iuser in tqdm(tmpdict):

        sentdata,idx_count=process_user(iuser,ds_rep_dict,symp_v,sim_v,topk,senti_rep_dict,senti_idx_dict,idx_count)

        u_twi_symlist[count] = sentdata[0]
        u_graph[count] = sentdata[1]
        u_gcngraph[count] = sentdata[2]
        count+=1

    pickle.dump(u_twi_symlist, foututs)
    foututs.close()
    pickle.dump(u_graph, foutug)
    foutug.close()
    pickle.dump(u_gcngraph, foutgg)
    foutgg.close()
    return idx_count,senti_rep_dict,senti_idx_dict



def get_bertout(sentence,voc_path='./all'):
    bert_sequence=Bert_representation.Get_bertout(sentence,voc_path)

    bert_sequence=torch.from_numpy(bert_sequence)
    bert_sequence=bert_sequence.reshape(1,len(bert_sequence))
    # print(bert_sequence.is_cuda)
    bert_out= Bert_encoder(bert_sequence.to(device))

    bert_rep=bert_out.last_hidden_state

    return bert_rep

def compute_cos(ten1,ten2):
    res=torch.nn.CosineSimilarity(dim=2, eps=1e-08)
    re=res(ten1,ten2)
    return re



def process_user(user_twi,ds_rep_dict,symp_v,sim_v,topk,idx_count):
    all_twi_len=len(user_twi['tweets'])
    sentdata = []

    twi_symlist_dic={}
    twi_symlist_set=[]
    twi_simlist_dic={}

    count=0


    twi_rep_set=[]
    twi_count=0
    for itw in user_twi['tweets']:
        # if twi_count>20:
        #     break
        textrep = get_bertout(itw['tweet_content']).mean(1).squeeze(1).tolist()
        twi_rep_set.append(textrep)
        twi_count+=1

    twi_count=0
    for itw in user_twi['tweets']:
        # if twi_count>20:
        #     break
        tmptext = itw['tweet_content']
        if tmptext == '':
            tmptext = '无。'


        #Sentence Symptom Knowledge
        textrep = twi_rep_set[count]
        textrep=torch.tensor(textrep).to(device)
        tmp_ds_sorce = compute_cos(textrep.repeat(1, ds_rep_dict.shape[1], 1), ds_rep_dict)

        values, indices = torch.topk(tmp_ds_sorce, k=topk)
        values = values.squeeze(1).squeeze(0)
        indices = indices.squeeze(1).squeeze(0)


        tmpsylist=[]
        for itk in range(0, len(indices)):
            if values[itk].item()>=symp_v:
                true_index=indices[itk].item()+1
                tmpsylist.append(true_index)
                if true_index not in twi_symlist_set:
                    twi_symlist_set.append(true_index)
        if len(tmpsylist)!=0:
            twi_symlist_dic[count] =tmpsylist


        #Sim set
        textrep = twi_rep_set[count]
        textrep=torch.tensor(textrep).to(device)
        twi_rep_set_tensor=torch.tensor(twi_rep_set).to(device).squeeze(1).unsqueeze(0)
        tmp_ss_sorce = compute_cos(textrep.repeat(1, twi_rep_set_tensor.shape[1], 1), twi_rep_set_tensor)
        if all_twi_len<topk:
            svalues, sindices = torch.topk(tmp_ss_sorce, k=all_twi_len)
            if all_twi_len==1:
                svalues = svalues.squeeze(1)
                sindices = sindices.squeeze(1)
            else :
                svalues = svalues.squeeze(1).squeeze(0)
                sindices = sindices.squeeze(1).squeeze(0)
        else:
            svalues, sindices = torch.topk(tmp_ss_sorce, k=topk)
            svalues = svalues.squeeze(1).squeeze(0)
            sindices = sindices.squeeze(1).squeeze(0)

        tmpsylist=[]
        for itk in range(0, len(sindices)):
            if svalues[itk].item()>=sim_v and sindices[itk].item() != count:
                tmpsylist.append(sindices[itk].item())
        if len(tmpsylist)!=0:
            twi_simlist_dic[count] =tmpsylist
        count+=1
        twi_count+=1

    #user twi
    all_twi_matrix = np.zeros((all_twi_len+len(twi_symlist_set), all_twi_len+len(twi_symlist_set))).astype('float32')

    for itwi in range(0,all_twi_len):
        all_twi_matrix[itwi][itwi] = 1
        if itwi<all_twi_len-1:
            all_twi_matrix[itwi+1][itwi] = 1
            all_twi_matrix[itwi][itwi+1] = 1

        if itwi in twi_symlist_dic.keys():
            for isym in twi_symlist_dic[itwi]:
                tmpsymindex = twi_symlist_set.index(isym) + all_twi_len
                if all_twi_matrix[tmpsymindex][tmpsymindex] != 1:
                    all_twi_matrix[tmpsymindex][tmpsymindex] = 1
                all_twi_matrix[itwi][tmpsymindex] = 1
                all_twi_matrix[tmpsymindex][itwi] = 1

        if itwi in twi_simlist_dic.keys():
            for isim in twi_simlist_dic[itwi]:
                all_twi_matrix[itwi][isim] = 1
                all_twi_matrix[isim][itwi] = 1
    #GCNgraph
    all_twi_graph_matrix = np.zeros((all_twi_len, all_twi_len)).astype('float32')

    for itwi in range(0,all_twi_len):
        all_twi_graph_matrix[itwi][itwi] = 1
        if itwi<all_twi_len-1:
            all_twi_graph_matrix[itwi+1][itwi] = 1
            all_twi_graph_matrix[itwi][itwi+1] = 1

    sentdata.append(twi_symlist_set)
    sentdata.append(all_twi_matrix)
    sentdata.append(all_twi_graph_matrix)

    return sentdata,idx_count




if __name__ == '__main__':
    ds_list=['情绪低落','对活动的兴趣减弱','难以集中注意力','无价值感','过度或不适当内疚','绝望','出现死亡或自杀的想法','食欲改变','睡眠改变','精神运动性躁动或迟钝','精力减少或疲劳']
    ds_exp_dict={
        '情绪低落': '”情绪低落“用于形容某人心情不佳、精神不振的状态。当一个人的情绪处于低落状态时，他们可能感到沮丧、悲伤、无助或缺乏动力和兴趣。',
        '对活动的兴趣减弱': '”对活动的兴趣减弱“指的是一个人对参与某种或多种活动的热情和兴趣降低了。这通常表现为一个人对曾经喜欢或热衷于的活动变得不再那么感兴趣，可能不再积极参与，甚至可能完全失去兴趣。',
        '难以集中注意力': '”难以集中注意力“指的是一个人无法将心思或思维有效地聚焦于某个任务、活动或信息上，导致注意力容易分散、思维不连贯或难以保持持久的专注力。',
        '无价值感': '”无价值感“是一个描述个体自我认知状态的短语，它指的是一个人认为自己没有价值或不值得被重视的感觉',
        '过度或不适当内疚': '”过度或不适当内疚“描述的是一种心理状态，其中个体对于某些事件或行为产生了一种超出正常范畴的内疚感，或者对于与自己无关或轻微责任的事件也感到内疚。这种内疚感可能并不基于实际的事实或合理的责任判断，而是源于个体对自己行为的过度解读或对他人的期望过于敏感。',
        '绝望': '”绝望“是一个深刻而复杂的情感状态，它通常指的是个体在面对无法改变或克服的困境时，感到彻底失去希望、意志和动力的心理状态。',
        '出现死亡或自杀的想法': '”出现死亡或自杀的想法“描述的是一种严重的心理状态，通常与极度的情感困扰、精神压力或心理健康问题有关。当个体经历极度痛苦、绝望、无助或其他强烈的负面情绪时，他们可能会开始思考死亡或自杀作为一种解脱或逃离现实的方式。',
        '食欲改变': '”食欲改变“是指个体的食欲与正常情况相比发生了显著的变化。这种变化可能表现为食欲不振，即食欲明显减退，对食物的兴趣降低，食量减少；或者食欲亢进，即食欲异常旺盛，食量明显增加。此外，食欲改变还可能包括嗜好习性的变化，比如对某种食物的偏好突然消失或出现新的偏好。',
        '睡眠改变': '“睡眠改变”指的是个体睡眠模式、睡眠质量或睡眠习惯与正常状态相比发生了显著的变化。这些变化可能包括入睡困难、睡眠浅、频繁醒来、早醒、睡眠时间减少或增多等。同时，也可能伴随着白天疲劳、注意力不集中、记忆力减退等睡眠不足的表现。',
        '精神运动性躁动或迟钝': '“精神运动性躁动或迟钝”是一种描述个体精神活动和身体运动状态异常的短语。精神运动性躁动指的是患者表现出过度的活动、兴奋和冲动，而精神运动性迟钝则是指患者表现出活动减少、反应迟缓的状态。',
        '精力减少或疲劳': '“精力减少或疲劳”描述的是一种身体和精神上的状态，其中个体感到缺乏活力、能量不足，以及持续的疲劳感。这种状态可能表现为在日常活动中感到力不从心，即使休息后也难以恢复精力。',
    }
    # ds_list_en=['depressed mood','diminished interest in activities','difficulty concentrating','feelings of worthlessness',
    #             'excessive or inappropriate guilt','hopelessness','thoughts of death or suicide',
    #             'changes in appetite','changes in sleep','psychomotor agitation or retardation','reduced energy or fatigue']
    # ds_exp_dict_en={
    #     'depressed mood': '“Depressed mood” is used to describe the state of someone who is in a bad mood and in low spirits. When a person is in a low mood, they may feel depressed, sad, helpless, or lack motivation and interest.',
    #     'diminished interest in activities': '“Diminished interest in activities” refers to a decrease in a person\'s enthusiasm and interest in participating in one or more activities. This usually manifests itself in the form of a person becoming less interested in an activity they once enjoyed or were passionate about, and may no longer actively participate in it, or may even lose interest altogether.',
    #     'difficulty concentrating': '“Difficulty concentrating” refers to a person\'s inability to focus his or her mind or thoughts effectively on a task, activity, or information, resulting in easy distraction, disjointed thinking, or difficulty in maintaining sustained concentration.',
    #     'feelings of worthlessness': '“Depressed” is a phrase that describes an individual\'s state of self-perception and refers to a person\'s feeling that he or she is worthless or unworthy of being valued.',
    #     'excessive or inappropriate guilt': '“Excessive or inappropriate guilt” is used to describe a psychological state in which an individual develops a sense of guilt about certain events or behaviours that is outside the realm of normalcy, or about events for which he or she is not involved or mildly responsible. Such feelings of guilt may not be based on actual facts or reasonable judgements of responsibility, but rather stem from the individual\'s overinterpretation of his or her own behaviour or over-sensitivity to the expectations of others.',
    #     'hopelessness': '“Hopelessness” is a deep and complex emotional state, and it usually refers to a psychological state in which an individual feels a complete loss of hope, will and motivation in the face of a predicament that cannot be changed or overcome.',
    #     'thoughts of death or suicide': '“Thoughts of death or suicide” is used to describe a serious psychological state that is usually associated with extreme emotional distress, mental stress, or mental health problems. When individuals experience extreme pain, despair, helplessness, or other intense negative emotions, they may begin to think about death or suicide as a way of relief or escape from reality.',
    #     'changes in appetite': '“Changes in appetite”It refers to a significant change in an individual\'s appetite from normal. This change may be manifested as loss of appetite, i.e., a marked loss of appetite, reduced interest in food, and a decrease in the amount of food eaten, or hyperphagia, i.e., an abnormally high appetite and a marked increase in the amount of food eaten. In addition, changes in appetite may include changes in habitual habits, such as the sudden disappearance of a preference for a particular food or the emergence of a new preference.',
    #     'changes in sleep': '“Changes in sleep” refers to significant changes in an individual\'s sleep patterns, sleep quality, or sleep habits compared to the normal state. These changes may include difficulty falling asleep, light sleep, frequent awakening, early waking, and decreased or increased sleep duration. It may also be accompanied by daytime fatigue, poor concentration, memory loss and other signs of sleep deprivation.',
    #     'psychomotor agitation or retardation': '“Psychomotor agitation or retardation” is a phrase that describes an abnormal state of an individual\'s mental activity and physical movement. Psychomotor agitation refers to a state in which the patient exhibits excessive activity, excitement, and impulsivity, while psychomotor retardation refers to a state in which the patient exhibits reduced activity and delayed responses.',
    #     'reduced energy or fatigue': '“Reduced energy or fatigue” is used to describe a physical and mental state in which the individual feels a lack of vitality, low energy, and a persistent sense of fatigue. This state may be manifested by feeling overwhelmed in daily activities and having difficulty regaining energy even after rest.',
    # }
    ds_rep=get_bertout(ds_exp_dict[ds_list[0]]).mean(1).unsqueeze(1)
    for ic in range(1,len(ds_list)):
        tmp_rep=get_bertout(ds_exp_dict[ds_list[ic]]).mean(1).unsqueeze(1)
        ds_rep=torch.cat((ds_rep,tmp_rep),1)

    # ds_rep_dict={}
    # ds_rep_dict[1] = ds_rep[:, 0:1, :].squeeze(0).squeeze(0).tolist()
    # for ikey in range(2,len(ds_list)+1):
    #     ds_rep_dict[ikey]=ds_rep[:,ikey-1:ikey,:].squeeze(0).squeeze(0).tolist()
    # # foutds = open('SYembedding_dict_BERT.pkl', 'wb')
    # # pickle.dump(ds_rep_dict, foutds)
    # # foutds.close()

    senti_idx_dict={}
    senti_rep_dict={}

    idx_count=1
    # senti_idx_dict
    # idx_count,senti_rep_dict,senti_idx_dict=processHG('./datasets/riskPredict/new/RP_train.json',ds_rep,0.85,0.86,3,senti_rep_dict,senti_idx_dict, idx_count)
    # idx_count,senti_rep_dict,senti_idx_dict=processHG('./datasets/riskPredict/new/RP_test.json',ds_rep,0.85,0.86,3,senti_rep_dict,senti_idx_dict, idx_count)
    # idx_count, senti_rep_dict, senti_idx_dict = processHG('./datasets/riskPredict/new/old_WU3D_train.json', ds_rep, 0.85, 0.86,
    #                                                       3, senti_rep_dict, senti_idx_dict, idx_count)
    # foutidx = open('SMembedding_idx_BERT_RP.pkl', 'wb')
    # pickle.dump(senti_idx_dict, foutidx)
    # foutidx.close()
    # foutsm = open('SMembedding_dict_BERT_RP.pkl', 'wb')
    # pickle.dump(senti_rep_dict, foutsm)
    # foutsm.close()




