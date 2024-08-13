# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
from torch.utils.data import DataLoader
from model4infer import SMHKEncoder,BERTEncoder,BERTLSTM,BERTGCN,BERTGAT
import Bert_representation
from DKHG_graph import process_user
import torch
import random
import math
import spacy
import argparse
from vocab import Vocab
import numpy as np
from transformers import BertModel
import os
import pickle
import _pickle as cPickle
import torch.nn.functional as F
from transformers import BertTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
nlp = spacy.load('zh_core_web_sm')

device = torch.device(device=1)

tokenizer = BertTokenizer.from_pretrained('./mlm/bert_finetune_sm_final')

bert4pro = BertModel.from_pretrained("./mlm/bert_finetune_sm_final")
Bert_encoder4pro = bert4pro  # Bert模型
Bert_encoder4pro.to(device)
Bert_encoder4pro.eval()

#---------------------------------process rawdata--------------------------------
def get_bertout(sentence,voc_path='./all'):
    bert_sequence=Bert_representation.Get_bertout(sentence,voc_path)

    bert_sequence=torch.from_numpy(bert_sequence)
    bert_sequence=bert_sequence.reshape(1,len(bert_sequence))
    # print(bert_sequence.is_cuda)
    bert_out= Bert_encoder4pro(bert_sequence.to(device))

    bert_rep=bert_out.last_hidden_state

    return bert_rep


ds_list = ['情绪低落', '对活动的兴趣减弱', '难以集中注意力', '无价值感', '过度或不适当内疚', '绝望',
           '出现死亡或自杀的想法', '食欲改变', '睡眠改变', '精神运动性躁动或迟钝', '精力减少或疲劳']
ds_exp_dict = {
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

ds_rep = get_bertout(ds_exp_dict[ds_list[0]]).mean(1).unsqueeze(1)
for ic in range(1, len(ds_list)):
    tmp_rep = get_bertout(ds_exp_dict[ds_list[ic]]).mean(1).unsqueeze(1)
    ds_rep = torch.cat((ds_rep, tmp_rep), 1)


#--------------------------------data-utlis-bert---------------------------------
def text_to_bert_sequence(text, max_len, padding="post", truncating="post"):
    text = tokenizer.tokenize(text)
    text = ["[CLS]"] + text + ["[SEP]"]
    sequence = tokenizer.convert_tokens_to_ids(text)
    return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)

def text_cps_bert_sequence(
        text, ckgsent, ckgsent_len, max_len, padding="post", truncating="post"
):
    text_raw_indices = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    text = "[CLS] " + text + " [SEP] "
    for si in ckgsent:
        text = text + si + " [SEP] "
    sequence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    sequence = pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)
    bert_segments_ids = np.asarray([0] * (len(text_raw_indices) + 2) + [1] * (ckgsent_len + 1))
    bert_segments_ids = pad_and_truncate(
        bert_segments_ids, max_len, padding=padding, truncating=truncating
    )
    return sequence, bert_segments_ids


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

#---------------------------------bert-Infer--------------------------------

def get_parameters(model, model_init_lr, multiplier):
    parameters = []
    enc_param_optimizer = list(model.named_parameters())
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in enc_param_optimizer if f'encoder.layer.{layer}.' in n],
            'lr': lr,
            'weight_decay': 0.0
        }
        parameters.append(layer_params)
        lr *= multiplier
    return parameters

class Inferer:
    def __init__(self, args):
        self.args = args

        #数据加载阶段
        # self.trainset = ABSADataset(args.dataset_file['train'],args)
        # self.testset = ABSADataset(args.dataset_file['test'],args)
        KGembedding_matrix_file_name = 'SYembedding_matrix_BERT.pkl'
        if os.path.exists(KGembedding_matrix_file_name):
            print('loading embedding_matrix:', KGembedding_matrix_file_name)
            SYembedding_matrix = pickle.load(open(KGembedding_matrix_file_name, 'rb'))
        else:
            SYembedding_matrix = np.zeros((12, 768))
            SYembedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(768), 1 / np.sqrt(768), (1, 768))
            SYembedding_matrix_file_name = 'SYembedding_dict_BERT.pkl'
            kgword_vec = load_KGword_vec(SYembedding_matrix_file_name)
            print('building embedding_matrix:', KGembedding_matrix_file_name)
            for i in range(1, len(kgword_vec) + 1):
                vec = kgword_vec[i]
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    SYembedding_matrix[i] = vec
            # pickle.dump序列化对象，将对象obj保存到文件file中去。
            pickle.dump(SYembedding_matrix, open(KGembedding_matrix_file_name, 'wb'))
        self.SYembedding_matrix = SYembedding_matrix


        #模型加载阶段ASGCN,DKF-BERT
        if args.model_name=='SM-HK-BERT':
            self.model = args.model_class(args,self.SYembedding_matrix)

            self.parameters = [p for p in self.model.parameters() if p.requires_grad]

            print('loading model {0} ...'.format(self.args.model_name))
            self.model.load_state_dict(torch.load(self.args.state_dict_path))
            #****
            bert_model = self.model.Bert_encoder
            optimizer_grouped_parameters = get_parameters(bert_model, args.bert_lr, 1)

        else:
            self.model = args.model_class(args)
            self.parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.model.to(args.device)
            self._print_args()
            #****
            bert_model = self.model.Bert_encoder
            bert_params_dict = list(map(id, bert_model.parameters()))
            base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
            optimizer_grouped_parameters = [
                {"params": base_params},
                {"params": bert_model.parameters(), "lr": args.bert_lr},
            ]

        self.optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg
        )

        self.global_f1 = 0.
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

        self.model = self.model
        self.model.to(args.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def processtext(self, iuserlist):
        all_data=[]
        for iuser in iuserlist:
            sentdata,_=process_user(iuser,ds_rep,0.85,0.86,3,[],[],1)
            u_twi_symlist = sentdata[0]
            u_graph= sentdata[1]
            u_gcngraph = sentdata[2]


            twi_text_bertlist = []

            twi_count = 0
            twi_maxlen=0
            for itw in iuser['tweets']:
                twi_text = itw['tweet_content'].strip()
                twilen=len(twi_text.split())
                if twilen>twi_maxlen:
                    twi_maxlen=twilen
                bert_text_sequence = text_to_bert_sequence(twi_text, args.max_len)
                twi_text_bertlist.append(bert_text_sequence)

                twi_count += 1
            print('twi_maxlen:',twi_maxlen)
            sym_list = u_twi_symlist
            sym_padding = [0] * (args.sy_max_len - len(sym_list))
            sym_add = np.array(sym_list + sym_padding)

            ugraph = np.array(u_graph)
            len_u_gra = len(ugraph)
            print("len_u_gra", len_u_gra)

            ugcngraph = np.array(u_gcngraph)
            len_u_gcngra = len(ugcngraph)


            list_pad = [0] * (args.max_len)
            list_pad = np.array(list_pad)

            for i in range(0, args.twi_max_len - len(twi_text_bertlist)):
                twi_text_bertlist.append(list_pad)

            data = {
                'twi_text_bert_sequence_list': np.array(twi_text_bertlist),
                'u_sym_list': sym_add,
                'u_twi_len': np.array(twi_count),
                'u_graph': np.pad(ugraph, (
                    (0, args.twi_max_len - len_u_gra), (0, args.twi_max_len - len_u_gra)), 'constant'),
                'u_gcngraph': np.pad(ugcngraph, (
                    (0, args.twi_max_len - len_u_gcngra), (0, args.twi_max_len - len_u_gcngra)), 'constant'),
            }
            all_data.append(data)
        return all_data


    def evaluate(self, user):
        pdata=self.processtext(user)
        pdata1 = DataLoader(dataset=pdata, batch_size=args.batch_size, shuffle=True,num_workers=2)
        for i_batch, sample_batched in enumerate(pdata1):
            inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
            outputs,_ = self.model(inputs)
            t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, default="state_dict/SM-HK-BERTRP2.pkl", help="DKF-BERTrest14,ASGCNrest14,")

    parser.add_argument('--model_name', default='SM-HK-BERT', type=str)
    parser.add_argument('--dataset', default='RP', type=str, help='')
    parser.add_argument("--hidden_dim", type=int, default=768, help="bert dim.")
    parser.add_argument("--num_class", type=int, default=3, help="Num of fsentiment class.")
    parser.add_argument("--aux_num_class", type=int, default=3, help="Num of fsentiment class.")

    # orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=2e-6, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=1e-5, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--num_epoch", type=int, default=25, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=10, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument('--device', default=1, type=str)
    parser.add_argument("--bert_out_dim", type=int, default=300)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    # lap14 1h1c:   1h2c:197  1h3c:223  1h4c:247 1h5c:270 1h6c:287 2h2c:358  2h3c:508
    # #rest14: 1h2c: 1h4c:199
    parser.add_argument("--max_len", type=int, default=12)
    parser.add_argument("--twi_max_len", type=int, default=9)
    parser.add_argument("--sy_max_len", type=int, default=9)
    # parser.add_argument("--sm_max_len", type=int, default=105)

    parser.add_argument('--repeat', default=1, type=int)

    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--GCNlayers', default=1, type=int)
    parser.add_argument('--kg_dropout', default=0.2, type=float)
    parser.add_argument('--aux_r', default=0.5, type=float)

    parser.add_argument('--symp_v', default=0.85, type=str)
    parser.add_argument('--sim_v', default=0.86, type=str)
    parser.add_argument('--topk', default=3, type=str)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument("--lower", default=True, help="Lowercase all words.")
    parser.add_argument("--direct", default=False)
    parser.add_argument("--loop", default=True)
    parser.add_argument("--reset_pooling", default=False, action="store_true")
    parser.add_argument("--output_merge", type=str, default="none", help="merge method to use, (none, gate)", )

    parser.add_argument('--type', default="1", type=str)
    args = parser.parse_args()

    model_classes = {
        'SM-HK-BERT': SMHKEncoder,
    }
    input_colses = {
        'SM-HK-BERT': [
            'twi_text_bert_sequence_list',
            'u_sym_list',
            'u_twi_len',
            'u_graph',
        ],
    }

    dataset_files = {
        'RP': {
            'train': './datasets/riskPredict/new/RP_train.json',
            'test': './datasets/riskPredict/new/RP_test.json'
        },
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.model_class = model_classes[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Inferer(args)

    #insert your user data
    isuer=[
    ]

    t_probs = ins.evaluate(isuer)
    print(t_probs.argmax(axis=-1)[0])

