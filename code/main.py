# coding=utf8
import time
import sys
import os

import numpy as np
import json
from tqdm import tqdm

import densenet
from PIL import Image

import torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn import metrics

# sys.path.append('../tools')
import parse, py_op
from glob import glob
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import traceback
from model import DenseNet121, Loss, CAT_LSTM

args = parse.args
import dataloader




def save_model(cnn, rnn):
    cnn_dict = cnn.state_dict()
    for key in cnn_dict.keys():
        cnn_dict[key] = cnn_dict[key].cpu()
    rnn_dict = rnn.state_dict()
    for key in rnn_dict.keys():
        rnn_dict[key] = rnn_dict[key].cpu()
    state_dict_all = {
            'cnn': cnn_dict,
            'rnn': rnn_dict,
            }
    torch.save( state_dict_all , '../file/cat_lstm.ckpt')

def healthy_image_embedding(cnn, train_loader):
    print("Collect healthy eyes' features")
    cnn.eval()
    demo_feat_dict = { }
    for i, (image, demos) in enumerate(tqdm(train_loader)):
        # if i > 20:
        #     break
        # print(image.size())
        image = Variable(image.cuda())
        feats = cnn(image)
        feats = feats.cpu().data.numpy()
        for feat, demo in zip(feats, demos):
            if demo not in demo_feat_dict:
                demo_feat_dict[demo] = []
            demo_feat_dict[demo].append(feat)
    return demo_feat_dict

def contrastive_attention(p_feat, h_feat_list):
    if len(h_feat_list) == 0:
        healthy_feat = np.zeros(len(p_feat), dtype=np.float32)
    else:
        np.random.shuffle(h_feat_list)
        weight_list = []
        for h_feat in h_feat_list[:100]:
            weight = np.dot(p_feat, h_feat) / np.sqrt(np.dot(p_feat, p_feat) * np.dot(h_feat, h_feat))
            # print(p_feat)
            # print(h_feat)
            # print(weight)
            # input()
            weight_list.append(weight)
        weight_list = np.array(weight_list)
        weight_list = np.exp(weight_list)
        assert len(weight_list) > 0
        assert min(weight_list) > 0
        weight_list = weight_list / np.sum(weight_list)
        # print(weight_list)
        assert weight_list.sum() > 0.99
        assert weight_list.sum() < 1.01
        healthy_feat = 0
        for w, feat in zip(weight_list, h_feat_list):
            healthy_feat += w * feat
        # print(healthy_feat.shape)
    return healthy_feat


def contrastive_feature(feats, demo_list, demo_feat_dict):
    # print('Function: contrastive_feature need to be written')
    bs, n_img, n_vec = feats.size()
    assert n_img == len(demo_list)
    healthy_feat_list = []
    feat_value = feats.cpu().data.numpy()
    for ib in range(bs):
        h_feat_list = []
        for ii in range(n_img):
            h_feat = contrastive_attention(feat_value[ib, ii], demo_feat_dict.get(demo_list[ii][ib], []))
            h_feat_list.append(h_feat)
        healthy_feat_list.append(h_feat_list)
    h_feat = Variable(torch.from_numpy(np.array(healthy_feat_list,np.float32)).cuda()).detach()
    c_feat = feats - h_feat
    # print(c_feat.size(), h_feat.size())
    return c_feat

def train_eval(epoch, cnn, rnn, train_loader, loss, optimizer, best_auc, demo_feat_dict, phase='train'):
    if 'train' in phase:
        cnn.train()
        rnn.train()
    else:
        cnn.eval()
        rnn.eval()
    loss_list, pred_list, label_list = [], [], []
    for i,data in enumerate(tqdm(train_loader)):

        demo_list = data[-1]
        data = data[:-1]
        images, stages, labels, timestamps = [Variable(x.cuda()) for x in data]
        # images: [bs, 8, 3, 512, 512] 
        # stages: [bs, 8, 67]
        # labels: [bs, 8]
        # timestamps: [bs, 8]

        size = list(images.size())                                                                          # [bs, 8, 3, 512, 512]
        images = images.view([size[0] * size[1], size[2], size[3], size[4]])
        feat = cnn(images).view([size[0], size[1], -1])                                                    # [bs, 8, 1024]
        c_feat = contrastive_feature(feat, demo_list, demo_feat_dict)
        # print('feature.size:', c_feat.size())

        probs, feat = rnn(c_feat, stages, timestamps)                                                       # Variable feat are used for clustering.
        # print('output.size:', probs.size())

        loss_output = loss(probs, labels, args.hard_mining)

        if 'train' in phase:
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()

        pred_list += list(probs.data.cpu().numpy().reshape(-1))
        label_list += list(labels.data.cpu().numpy().reshape(-1))
        loss_list.append(loss_output.data.cpu().numpy())
            

    new_pred, new_label = [], []
    for p, l in zip(pred_list, label_list):
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
        save_model(cnn, rnn)
    # print(phase, epoch, loss, auc, int(np.sum(label_list)), len(label_list) - int(np.sum(label_list)), np.mean(label_list), best_auc[0], best_auc[1])
    print('Phase: {:s}      Epoch: {:d}      Loss: {:1.4f}    AUC: {:1.4f}        P/N:{:d}/{:d}/{:1.5f}         Best AUC: {:1.4f}/{:d}'.format(phase, epoch, loss, auc, int(np.sum(label_list)), len(label_list) - int(np.sum(label_list)), np.mean(label_list), best_auc[0], best_auc[1]))
    return best_auc

def main():
    pid_side_tid_file_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'))
    cudnn.benchmark = True
    cnn = DenseNet121().cuda()
    rnn = CAT_LSTM().cuda()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=args.lr)
    cnn = torch.nn.DataParallel(cnn).cuda()
    rnn = torch.nn.DataParallel(rnn).cuda()
    loss = Loss().cuda()

    if args.phase == 'train':
        inference_dataset = dataloader.InferenceDataSet(phase='train')
        inference_loader = DataLoader( dataset=inference_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
        train_dataset  = dataloader.PredictionDataSet(pid_side_tid_file_dict, phase='train')
        train_loader = DataLoader( dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        valid_dataset  = dataloader.PredictionDataSet(pid_side_tid_file_dict, phase='valid')
        valid_loader = DataLoader( dataset=valid_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        demo_feat_dict = { }

        best_auc =[0, 0] 
        for epoch in range(100):

            args.epoch = epoch

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

            train_eval(epoch, cnn, rnn, train_loader, loss, optimizer, best_auc, demo_feat_dict, 'train')
            best_auc = train_eval(epoch, cnn, rnn, valid_loader, loss, optimizer, best_auc, demo_feat_dict, 'valid')
            demo_feat_dict = healthy_image_embedding(cnn, inference_loader)


if __name__ == '__main__':
    main()
