# coding=utf8

import time
import sys
import os

import numpy as np
import json
from tqdm import tqdm

import densenet
import torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

# sys.path.append('../tools')
import parse, py_op
from glob import glob

args = parse.args



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, output, target, use_hard_mining=False):
        output = self.sigmoid(output)

        # hard_mining 
        output = output.view(-1)
        target = target.view(-1)

        index = target > -0.5
        output = output[index]
        target = target[index]

        # pos
        pos_index = target > 0.5
        pos_output = output[pos_index]
        pos_target = target[pos_index]
        if len(pos_output):
            num_hard_pos = max(len(pos_output)/4, min(5, len(pos_output)))
            if use_hard_mining and len(pos_output) > 5:
                pos_output, pos_target = hard_mining(pos_output, pos_target, num_hard_pos, largest=False)
            pos_loss = self.classify_loss(pos_output, pos_target) * 0.5
        else:
            pos_loss = 0


        # neg
        neg_index = target < 0.5
        neg_output = output[neg_index]
        neg_target = target[neg_index]
        if len(neg_output):
            if use_hard_mining:
                num_hard_neg = len(pos_output) * 2
                neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard_neg, largest=True)
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5
        else:
            neg_loss = 0

        loss = pos_loss + neg_loss

        return loss


def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels




class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.inplanes = 1024
        self.densenet121 = densenet.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # self.classifier = nn.Linear(num_ftrs, out_size)

    def forward(self, x, phase='train'):
        feats = self.densenet121(x)     # (32, 1024, 2, 16)
        out = self.conv(feats) # (32, 1024, 8, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return out


class CAT_LSTM(nn.Module):
    def __init__(self):
        super(CAT_LSTM, self).__init__()
        self.lstm = TLSTM(input_size = 1024 * 2, hidden_size=256)
        self.classifier = nn.Sequential(
                # nn.Linear(1024 * 2, 1024), 
                # nn.ReLU(),
                # nn.Linear(1024, 1024), 
                # nn.ReLU(),
                nn.Linear(256, 1), 
                )
        self.stage_embedding = nn.Sequential(
                nn.Linear(67, 256), 
                nn.ReLU(),
                nn.Linear(256, 512), 
                nn.ReLU(),
                nn.Linear(512, 1024), 
                nn.ReLU(),
                )
    def forward(self, x, stages, timestamps):
        size = stages.size()
        stages = stages.view((size[0] * size[1], size[2]))
        stages = self.stage_embedding(stages)
        stages = stages.view((size[0], size[1], -1))


        x = torch.cat((x, stages), 2)
        out = self.lstm(x, timestamps)
        size = out.size()
        out = out.contiguous().view((size[0] * size[1], size[2]))
        feat = out
        out = self.classifier(out)
        return out, feat
        
        
        
class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=True):
        # assumes that batch_first is always true
        super(TLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs
