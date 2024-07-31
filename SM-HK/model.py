# -*- coding: utf-8 -*-
# coding=utf-8
# coding:utf-8
import sys

sys.path.append('../')
import torch

from transformers import BertModel
import torch.nn.functional as F
from torch import nn

bert = BertModel.from_pretrained("./mlm/bert_finetune_sm_final")

class SMHKEncoder(nn.Module):
    def __init__(self, args,SYembedding_matrix):
    # def __init__(self, args):

        super().__init__()
        self.args = args
        self.classifier = nn.Linear(args.bert_out_dim , args.num_class)  # 分类器t
        self.classifier2= nn.Linear(args.bert_out_dim , args.aux_num_class)  # 分类器t


        self.Bert_encoder = bert#Bert模型

        self.in_drop = nn.Dropout(args.bert_dropout)  # Twitter  bert输出时第一次dropout

        self.textdense = nn.Linear(args.hidden_dim, args.bert_out_dim)  #线性层 隐藏层维度到bert维度
        self.sydense = nn.Linear(args.hidden_dim, args.bert_out_dim)  #线性层 隐藏层维度到bert维度

        #KG模块

        self.SYembed = nn.Embedding.from_pretrained(torch.tensor(SYembedding_matrix, dtype=torch.float))
        self.sy_embed_dropout = nn.Dropout(args.kg_dropout)


        self.kggat1=B_GAT(args.bert_out_dim, args.bert_out_dim,args.kg_dropout,0.2,args.nhead,args.batch_size)
        # self.kggat2=B_GAT(args.bert_out_dim, args.bert_out_dim,args.kg_dropout,0.2,args.nhead,args.batch_size)
        self.dense4gat = nn.Linear(args.nhead*args.bert_out_dim, args.bert_out_dim)  #线性层 隐藏层维度到bert维度


        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)

        self.fcs = nn.Sequential(nn.Linear(args.bert_out_dim, args.bert_out_dim), nn.ReLU(),
                                 nn.Linear(args.bert_out_dim, args.bert_out_dim), nn.ReLU())



    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def merge_tensor(self,bert_out,text_len,cprep):
        tm1=torch.IntTensor([bert_out.shape[1]]).repeat(bert_out.shape[0]).to(bert_out.device)
        st_len = tm1 - text_len
        batch_size, maxlen=bert_out.shape[0],bert_out.shape[1]
        res_ten=bert_out[0:1,:text_len[0],:]
        res_cp_ten=cprep[0:1,:st_len[0],:]
        res_ten=torch.cat((res_ten,res_cp_ten),1)
        for i in range(1,batch_size):
            tmp_res_ten = bert_out[i:i+1, :text_len[i], :]
            tmp_res_cp_ten = cprep[i:i+1, :st_len[i], :]
            tmp_res_ten = torch.cat((tmp_res_ten, tmp_res_cp_ten), 1)
            res_ten=torch.cat((res_ten,tmp_res_ten),0)
        return res_ten


    def retail_text(self,bert_out,text_len):
        batch_size, maxlen=bert_out.shape[0],bert_out.shape[1]
        tm1=torch.IntTensor([bert_out.shape[1]]).repeat(bert_out.shape[0]).to(bert_out.device)
        pad_len = tm1 - text_len
        res_ten=bert_out[0:1,:text_len[0],:]
        pad_ten=torch.IntTensor(1, pad_len[0],bert_out.shape[2]).zero_().to(bert_out.device)
        res_ten=torch.cat((res_ten,pad_ten),1)
        for i in range(1,batch_size):
            tmp_res_ten = bert_out[i:i+1, :text_len[i], :]
            tmp_pad_ten = torch.IntTensor(1, pad_len[i], bert_out.shape[2]).zero_().to(bert_out.device)
            tmp_res_ten = torch.cat((tmp_res_ten, tmp_pad_ten), 1)
            res_ten=torch.cat((res_ten,tmp_res_ten),0)
        return res_ten


    def forward(self, inputs):
        twi_text_bert_sequence_list,\
        u_sym_list,\
        u_twi_len,\
        u_graph, \
            = inputs

        #twi-level
        batch_num=twi_text_bert_sequence_list.shape[0]
        all_twi_re =torch.randn(1,1,1)
        catflag=1
        for ibatch in range(0,batch_num):

            twi_text_out, _ = self.Bert_encoder(twi_text_bert_sequence_list[ibatch:ibatch+1,:,:].squeeze(0),return_dict=False)  # bert twit输出
            twi_text_out = self.in_drop(twi_text_out)  # bert输出时第一次dropout
            twi_text_out = self.textdense(twi_text_out)

            allasbrep1 = twi_text_out.mean(1).unsqueeze(0) #1*300

            if catflag==1:
                all_twi_re=allasbrep1
                catflag=0
            else:
                all_twi_re=torch.cat((all_twi_re, allasbrep1), 0)

        #all_user
        syrep = self.SYembed(u_sym_list)
        syrep = self.sy_embed_dropout(syrep)
        syrep = self.sydense(syrep)

        # cat rep
        gat_input = self.merge_tensor(all_twi_re, u_twi_len, syrep)  #batchsize*160*300

        #GAT输入
        xgat = F.relu(self.kggat1(gat_input, u_graph))
        xgat =self.dense4gat(xgat)

        graph_outputs=(xgat.sum(1, keepdim=True)).squeeze(1)

        cat_outputs = self.fcs(graph_outputs)
        logits = self.classifier(cat_outputs)
        logits2 = self.classifier2(cat_outputs)


        return logits,logits2  # 2分类向量



class B_GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    # 对实例的属性进行初始化
    def __init__(self, in_features, out_features, dropout, alpha, batchsize,concat=True):
        super(B_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # 学习因子
        self.concat = concat  # 定义是否进行concat操作
        # 定义W值，其维度就是输入的维度和输出的维度(1433*8)
        self.W = nn.Parameter(torch.empty(size=(batchsize,in_features, out_features)))
        # 初始化W向量，服从均匀分布
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(batchsize,2 * out_features, 1)))
        # 对α进行初始化
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # 进行leakyrelu操作，将LeakyRelu中的学习率设置为alpha?
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.bmm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self.B_prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)  # 建立了一个极小值矩阵
        # torch.where(condition，a，b)
        # 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵有值的点变成eij,没有值的点变成无穷小的一个值,2708*2708维
        attention = F.softmax(attention, dim=1)  # 即对相连接的节点之间求了一个attention系数,每行的attention系数加起来正好是1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数,2708*1433

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def B_prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # Wh即W和h向量相乘
        ab=self.a[:,:self.out_features, :]
        af=self.a[:,self.out_features:, :]
        Wh1 = torch.matmul(Wh, self.a[:,:self.out_features, :])  # 8*1
        Wh2 = torch.matmul(Wh, self.a[:,self.out_features:, :])  # 8*1,取了各自向量对应的权重，所以Wh1和Wh2不一样
        # broadcast add
        # 通过广播机制完成这个求注意力机制
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class B_GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads,batchsize):
        """Dense version of GAT."""
        super(B_GAT, self).__init__()
        self.dropout = dropout
        #这句话即多头的计算循环代码,参数分别是输入的特征1433维，中间隐层的特征8维,即8个隐层单元，
        #得到的结果是8个attention layer和一个拼接在一起的attentions
        self.attentions = [B_GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, batchsize=batchsize,concat=True) for _ in range(nheads)]
        #将8个头分别赋予到add中，怀疑这里是为下面的x=torch.cat中的循环做准备
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        # x就是train函数传入的特征
        x = F.dropout(x, self.dropout, training=self.training)
        #对每一个attention layer 进行拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


