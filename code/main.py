# coding=utf8
import time
import sys
import os

import numpy as np
import dataloader
import json
from tqdm import tqdm

import densenet
import resnet
from PIL import Image

import torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn import metrics

sys.path.append('../tools')
import parse, py_op
from glob import glob
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import traceback

args = parse.args



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, output, target, use_hard_mining=False):
        output = self.sigmoid(output)


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

def save_model(save_dir, phase, name, epoch, f1score, model):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, phase)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    state_dict_all = {
            'state_dict': state_dict,
            'epoch': epoch,
            'f1score': f1score,
            }
    torch.save( state_dict_all , os.path.join(save_dir, '{:s}.ckpt'.format(name)))
    if 'best' in name and f1score > 0.3:
        torch.save( state_dict_all , os.path.join(save_dir, '{:s}_{:s}.ckpt'.format(name, str(epoch))))


def train_eval(epoch, model, train_loader, loss, optimizer, best_auc, phase='train'):
    # print '\n',epoch, phase
    cnn, rnn = model
    if 'train' in phase:
        cnn.train()
        rnn.train()
    else:
        cnn.eval()
        rnn.eval()
    loss_list, pred_list, label_list = [], [], []

    result_list = []
    feat_list = []
    for i,data in enumerate(tqdm(train_loader)):

        # if i >= 200:
        #     break
        pid_tid_list = data[-1]
        data = data[:-1]

        images, stages, labels, grp_feat = [Variable(x.cuda()) for x in data][:4]

        size = list(images.size()) # [4,7,3, 256, 256]
        images = images.view([size[0] * size[1], size[2], size[3], size[4]])

        features = cnn(images)
        features = features.view((size[0], size[1], features.size(1)))
        probs, feat = rnn(features, stages)
        feat = feat.data.cpu().numpy()
        feat_list.append([pid_tid_list, feat, labels.data.cpu().numpy()])
        if args.use_cl:
            feat = feat - grp_feat

        # print('probs', probs.size())
        # print('labels', labels.size())
        result_list.append([pid_tid_list, probs.data.cpu().numpy(), labels.data.cpu().numpy()])

        loss_output = loss(probs, labels, args.hard_mining)
        try:

            if 'train' in phase:
                optimizer.zero_grad()
                loss_output.backward()
                optimizer.step()

            pred_list += list(probs.data.cpu().numpy().reshape(-1))
            label_list += list(labels.data.cpu().numpy().reshape(-1))
            loss_list.append(loss_output.data.cpu().numpy())
        except:
            pass
            
        # 保存中间结果到 data/middle_result，用于分析
        if i == 0:
            images = images.data.cpu().numpy() * 128 + 128
            for ii in range(len(images)):
                middle_dir = os.path.join(args.file_dir, 'middle_result')
                if not os.path.exists(middle_dir):
                    os.mkdir(middle_dir)
                middle_dir = os.path.join(middle_dir, phase)
                if not os.path.exists(middle_dir):
                    os.mkdir(middle_dir)
                Image.fromarray(images[ii].astype(np.uint8).transpose(1,2,0)).save(os.path.join(middle_dir, str(ii)+'.image.png'))

    new_pred, new_label = [], []
    for p,l in zip(pred_list, label_list):
        if l >= 0:
            new_pred.append(p)
            new_label.append(l)
    label_list = new_label
    pred_list = new_pred
    fpr, tpr, thr = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)
    loss = np.array(loss_list).mean()

    if phase == 'valid' and auc > best_auc[0]:
        best_auc = [auc, epoch]
        torch.save(result_list, '../file/result.ckpt')
        if not args.use_cl:
            torch.save(feat_list, '../file/feat.ckpt')
    # print(phase, epoch, loss, auc, int(np.sum(label_list)), len(label_list) - int(np.sum(label_list)), np.mean(label_list), best_auc[0], best_auc[1])
    print('Phase: {:s}      Epoch: {:d}      Loss: {:1.4f}    AUC: {:1.4f}        P/N:{:d}/{:d}/{:1.5f}         Best AUC: {:1.4f}/{:d}'.format(phase, epoch, loss, auc, int(np.sum(label_list)), len(label_list) - int(np.sum(label_list)), np.mean(label_list), best_auc[0], best_auc[1]))
    return best_auc



class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
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
        self.classifier = nn.Linear(num_ftrs, out_size)

    def forward(self, x, phase='train'):
        feats = self.densenet121(x)     # (32, 1024, 2, 16)
        out = self.conv(feats) # (32, 1024, 8, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM ( input_size=1024 * 2,
                              hidden_size=256,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=False)
        self.classifier = nn.Sequential(
                # nn.Linear(1024 * 2, 1024), 
                # nn.ReLU(),
                # nn.Linear(1024, 1024), 
                # nn.ReLU(),
                nn.Linear(256, 1), 
                )
        self.stage_embedding = nn.Sequential(
                nn.Linear(12, 256), 
                nn.ReLU(),
                nn.Linear(256, 512), 
                nn.ReLU(),
                nn.Linear(512, 1024), 
                nn.ReLU(),
                )
    def forward(self, x, stages):
        size = stages.size()
        stages = stages.view((size[0] * size[1], size[2]))
        stages = self.stage_embedding(stages)
        stages = stages.view((size[0], size[1], -1))

        # x_pre = x.data.cpu()
        # x_pre[:, 1:] = x_pre[:, :-1]
        # x_pre[:, 0] = 0
        # x_d = Variable((x.data.cpu() - x_pre).cuda())
        # x = torch.cat((x, x_d, stages), 2)

        x = torch.cat((x, stages), 2)
        out, _ = self.lstm(x)
        # out = out[:, -1, :]
        # out = torch.cat((out, stages), 2)
        size = out.size()
        # print(out.size())
        out = out.contiguous().view((size[0] * size[1], size[2]))
        # print(out.size())
        feat = out
        out = self.classifier(out)
        return out, feat

def main():

    pid_side_tid_file_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'))

    cudnn.benchmark = True
    cnn = DenseNet121(1).cuda()
    rnn = LSTM()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=args.lr)
    cnn = torch.nn.DataParallel(cnn).cuda()
    # rnn = torch.nn.DataParallel(rnn).cuda()
    rnn = rnn.cuda()
    loss = Loss().cuda()
    model = [cnn, rnn]


    if args.phase == 'train':
        # train_dataset  = dataloader.DetectionDataSet(pid_side_tid_file_dict, phase='train')
        train_dataset  = dataloader.CLPredictionDataSet(pid_side_tid_file_dict, phase='train')
        train_loader = DataLoader( dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        # valid_dataset  = dataloader.DetectionDataSet(pid_side_tid_file_dict, phase='valid')
        valid_dataset  = dataloader.CLPredictionDataSet(pid_side_tid_file_dict, phase='valid')
        valid_loader = DataLoader( dataset=valid_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

        best_auc =[0, 0] 
        for epoch in range(8):

            args.epoch = epoch

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

            train_eval(epoch, model, train_loader, loss, optimizer, best_auc, 'train')
            best_auc = train_eval(epoch, model, valid_loader, loss, optimizer, best_auc, 'valid')


    



if __name__ == '__main__':
    main()
