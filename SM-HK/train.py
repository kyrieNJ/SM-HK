# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
import sys
from torch.utils.data import DataLoader
from model import SMHKEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import random
import math
import argparse
import numpy as np
import torch.nn as nn
from transformers import BertModel

from sklearn import metrics
from vocab import Vocab
from data_utils import PDUDDataset
from tqdm import tqdm
import pickle
import torch.utils.data.distributed


record_lsit=[]

class Instructor:
    def __init__(self, args):
        self.args = args

        #数据加载阶段
        self.trainset = PDUDDataset(args.dataset_file['train'],args)
        self.testset = PDUDDataset(args.dataset_file['test'],args)
        #模型训练阶段
        self.model = args.model_class(args,self.trainset.SYembedding_matrix)
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


        self.optimizer = self.args.optimizer(
            optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.l2reg
        )

        self.global_acc = 0.
        self.global_im = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

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

    def _train(self, criterion,criterion2, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0

        for epoch in range(self.args.num_epoch):
            print('-' * 100)
            print('epoch: ', epoch)
            n_correct, n_total,loss_total = 0, 0,0
            increase_flag = False
            self.model.train()  # 训练过程

            for i_batch, sample_batched in tqdm(enumerate(self.train_data_loader)):
            # for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                # for i in inputs:
                #     print(i)
                outputs,outputs2 = self.model(inputs)
                targets = sample_batched['risk_label'].to(self.args.device)

                #aux
                aux_targets=sample_batched['sm_lable'].to(self.args.device)
                loss1 = criterion(outputs, targets)
                loss2 = criterion2(outputs2, aux_targets)
                loss=loss1+self.args.aux_r*loss2

                #wo aux
                # loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                if global_step % self.args.log_step == 0:
                    train_loss = loss_total / n_total
                    train_acc = n_correct / n_total
                    # print('train_acc:',train_acc)
                    print("train loss: {:.4f}, train acc: {:.4f}, ".format(train_loss, train_acc))
                    #       "valid explicit acc: {:.4f}, valid implicit acc: {:.4f} ".format(explicit_acc, implicit_acc))


            print("开始测试：")
            test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)#评估过程
            if test_f1 > max_test_f1:
                all_max_test_acc=test_acc

                max_test_acc = test_acc
                max_test_f1 = test_f1
                # record_lsit.append((max_test_acc,max_test_f1,max_test_expacc,max_test_impacc))
                increase_flag = True
                # torch.save(self.model.state_dict(), 'state_dict/' + self.args.model_name + self.args.dataset + '2.pkl')

                print('>>> best model saved.')

            print('loss: {:.4f}, acc: {:.4f}, \n test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))


            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 8:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0

        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in tqdm(enumerate(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                t_targets = t_sample_batched['risk_label'].to(self.args.device)
                t_outputs,_= self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)


                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1,2], average='macro')

        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion2=nn.CrossEntropyLoss()


        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=True,num_workers=2,drop_last=True)
        self.test_data_loader = DataLoader(dataset=self.testset, batch_size=args.batch_size, shuffle=False,num_workers=2,drop_last=True)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.args.model_name + '_' + self.args.dataset + '_val_RP.txt', 'w', encoding='utf-8')

        max_acc = 0
        max_f1 = 0

        for i in range(args.repeat):
            print('repeat: ', (i+1))
            f_out.writelines('repeat: '+str(i+1))
            self._reset_params()
            max_test_acc, max_test_f1 = self._train(criterion,criterion2, self.optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}        '.format(max_test_acc, max_test_f1))
            f_out.writelines('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))

            if(max_acc <= max_test_acc or max_f1 <= max_test_f1 ):
                max_acc = max_test_acc
                max_f1 = max_test_f1
            print('-' * 100)


        print("max_acc:", max_acc)
        print("max_f1:", max_f1)
        f_out.writelines('------------------------------------------------------------------------\n')
        f_out.writelines('max_acc: {0}, max_f1: {1}\n'.format(max_acc, max_f1))
        f_out.writelines('------------------------------------------------------------------------\n')
        for i in record_lsit:
            print(i)
        f_out.close()

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

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SM-HK-BERT', type=str)
    parser.add_argument('--dataset', default='RP', type=str, help='')
    parser.add_argument("--hidden_dim", type=int, default=768, help="bert dim.")
    parser.add_argument("--num_class", type=int, default=3, help="Num of fsentiment class.")
    parser.add_argument("--aux_num_class", type=int, default=3, help="Num of fsentiment class.")


    #orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=2e-6, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=1e-5, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--num_epoch", type=int, default=25, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=10, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--bert_out_dim", type=int, default=300)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    # lap14 1h1c:   1h2c:197  1h3c:223  1h4c:247 1h5c:270 1h6c:287 2h2c:358  2h3c:508
    # #rest14: 1h2c: 1h4c:199
    parser.add_argument("--max_len", type=int, default=75)
    parser.add_argument("--twi_max_len", type=int, default=160)
    parser.add_argument("--sy_max_len", type=int, default=160)
    # parser.add_argument("--sm_max_len", type=int, default=105)


    parser.add_argument('--repeat', default=1, type =int)

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
    parser.add_argument("--output_merge",type=str,default="none",help="merge method to use, (none, gate)",)

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
        'RP_en': {
            'train': './datasets/riskPredict/new_en/RP_en_train.json',
            'test': './datasets/riskPredict/new_en/RP_en_test.json'
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


    ins = Instructor(args)
    ins.run()

