# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
import numpy as np
import os
import json
from tqdm import tqdm
# import skimage
# from skimage import io
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from torch.utils.data import Dataset
import time
import torchvision.transforms as transforms

import sys
sys.path.append('../tools')
import parse, py_op
args = parse.args

transform = transforms.Compose([
    transforms.Pad(100),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(256),
    transforms.ToTensor()])



class PredictionDataSet(Dataset):
    def __init__(self, 
            pid_side_tid_file_dict,
            phase='train',          # phase
            ):
        super(PredictionDataSet, self).__init__()

        image_list, label_list, pid_tid_side = [], [], []
        for pid in pid_side_tid_file_dict[phase]:
            for side in pid_side_tid_file_dict[pid]:
                tids = sorted(pid_side_tid_file_dict[pid][side].keys(), key=lambda s:int(s))
                stgs = [pid_side_tid_file_dict[pid][side][t][0] for t in tids]

                ltamd = [-1, -1]
                for i,t,s in zip(range(len(tids)), tids, stgs):
                    if s > 9:
                        ltamd = [i, t]
                        break

                pts = []
                for i,t,s in zip(range(len(tids)), tids, stgs):
                    # if ltamd[0] < 0 or (i < ltamd[0] - 1 and int(t) < int(ltamd[1]) - 2):
                    stage = pid_side_tid_file_dict[pid][side][str(t)][0]
                    if ltamd[0] == i + 1:
                        file_list = pid_side_tid_file_dict[pid][side][str(t)]
                        pts.append([pid, t, side, 1, stage])
                        label_list.append(1)
                        break
                    elif ltamd[0] < 0 or int(t) < int(ltamd[1]) - 10:
                        if int(tids[-1]) - int(t) < 6:
                            break
                            # file_list = pid_side_tid_file_dict[pid][side][str(t)]
                            # pts.append([pid, t, side, -1, stage])
                            # label_list.append(-1)
                        else:
                            file_list = pid_side_tid_file_dict[pid][side][str(t)]
                            pts.append([pid, t, side, 0, stage])
                            label_list.append(0)
                    else:
                        file_list = pid_side_tid_file_dict[pid][side][str(t)]
                        if file_list[0] > 9:
                            break
                        pts.append([pid, t, side, 1, stage])
                        label_list.append(1)
                if len(pts):
                    pid_tid_side.append(pts)

        self.phase = phase
        self.pid_tid_side = pid_tid_side
        self.pid_side_tid_file_dict = pid_side_tid_file_dict
        self.pid_demo_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_demo_dict.json'))
        label_list = [l for l in label_list if l >= 0]
        print('Positive rate: {:1.3f}/{:d}'.format(np.mean(label_list), len(label_list)))

    def stage_vector(self, tl):
        assert len(tl) > 0
        # print(tl)
        v = [tl[0][1]]
        for i in range(1, len(tl)):
            gap = tl[i][0] - tl[i-1][0]
            assert gap >= 1
            for _ in range(gap - 1):
                v.append(tl[i-1][1])
            v.append(tl[i][1])
        ns = 12
        while len(v) < ns:
            v = [0]  + v
        v = v[- ns:]
        assert max(v) <= 9
        return v


        for i in range(1, len(tl)):
            if tl[i][1] < tl[i-1][1]:
                tl[i][1] = tl[i-1][1]

        t_list = [t[0] for t in tl]
        s_list = [t[1] for t in tl]
        t_end = dict()
        for t,s in zip(t_list, s_list):
            t_end[s] = t
        stage_set = set([t[1] for t in tl])



        p = np.zeros((10, 10))
        for i in range(10):
            # v.append(s_cnt.get(i, 0))
            if i in stage_set:
                ti = t_list[s_list.index(i)]
                tj = t_end[i]
                # v.append(tj - ti)
                p[i, i] = tj - ti
                for ii in range(i + 1):
                    for jj in range(i, 10):
                        p[ii, jj] = max(p[ii, jj], p[i,i])
            for j in range(i+1, 10):
                if i in stage_set and j in stage_set:
                    ti = t_list[s_list.index(i)]
                    tj = t_list[s_list.index(j)]
                    # v.append(tj - ti)
                    p[i, j] = tj - ti
                    for ii in range(i + 1):
                        for jj in range(j, 10):
                            p[ii, jj] = max(p[ii, jj], p[i,j])
        for i in range(10):
            for j in range(i, 10):
                v.append(p[i,j])
        # print(len(v))
        # print(len(v), v)
        return v

    def __getitem__(self, index):
        pts = self.pid_tid_side[index]
        assert len(pts) > 0
        n_img = 8
        img_list = []
        labels = []
        vectors = []
        if len(pts) > n_img:
            pts = pts[ - n_img:]
        tl = []
        pid_tid_list = [pts[0][0] + ' ' + pts[0][2]]
        for pid, tid, side, label, stage in pts:
            img_list.append(self.read_image(pid, tid, side))
            # img_list.append(torch.from_numpy(np.zeros([3, 256,256])))
            # labels.append(label)
            try:
                enrollage = self.pid_demo_dict[pid]['enroll_age']
                age = float(enrollage) + int(tid) / 2.0
            except:
                age = 50
            if age <= 75:
                # labels.append(label - 100)
                labels.append(label)
            else:
                labels.append(label)
            tl.append([int(tid), int(stage)])
            vectors.append(self.stage_vector(tl))
            pid_tid_list.append(tid)




        if len(img_list) < n_img:
            img = torch.from_numpy(np.zeros((3, 256, 256), dtype=np.float32))
            for _ in range(n_img - len(pts)):
                img_list.append(img)
                labels.append(-1)
                vectors.append(vectors[-1])
                pid_tid_list.append('-1')
        labels_all = np.array(labels, dtype=np.float32)
        labels_all[labels_all < -50] += 100
        return torch.stack(img_list), np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.float32), labels_all, pid_tid_list

    def read_image(self, pid, tid, side):
        image_name = self.pid_side_tid_file_dict[pid][side][tid][1:]
        np.random.shuffle(image_name)
        image_name = image_name[0]
        # print(image_name)
        image = Image.open(image_name).convert('RGB')
        h,w = image.size
        h = int(h/2)
        w = int(w/2)
        image = image.resize((h,w))
        image = transform(image)
        return image

    def __len__(self):
        return len(self.pid_tid_side) 

class CLPredictionDataSet(Dataset):
    def __init__(self, 
            pid_side_tid_file_dict,
            phase='train',          # phase
            ):
        super(CLPredictionDataSet, self).__init__()

        image_list, label_list, pid_tid_side = [], [], []
        self.pid_demo_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_demo_dict.json'))
        for pid in pid_side_tid_file_dict[phase]:
            if pid not in self.pid_demo_dict:
                continue
            for side in pid_side_tid_file_dict[pid]:
                tids = sorted(pid_side_tid_file_dict[pid][side].keys(), key=lambda s:int(s))
                stgs = [pid_side_tid_file_dict[pid][side][t][0] for t in tids]

                ltamd = [-1, -1]
                for i,t,s in zip(range(len(tids)), tids, stgs):
                    if s > 9:
                        ltamd = [i, t]
                        break

                pts = []
                for i,t,s in zip(range(len(tids)), tids, stgs):
                    # if ltamd[0] < 0 or (i < ltamd[0] - 1 and int(t) < int(ltamd[1]) - 2):
                    stage = pid_side_tid_file_dict[pid][side][str(t)][0]
                    if ltamd[0] == i + 1:
                        file_list = pid_side_tid_file_dict[pid][side][str(t)]
                        pts.append([pid, t, side, 1, stage])
                        label_list.append(1)
                        break
                    elif ltamd[0] < 0 or int(t) < int(ltamd[1]) - 8:
                        if int(tids[-1]) - int(t) < 6:
                            break
                            # file_list = pid_side_tid_file_dict[pid][side][str(t)]
                            # pts.append([pid, t, side, -1, stage])
                            # label_list.append(-1)
                        else:
                            file_list = pid_side_tid_file_dict[pid][side][str(t)]
                            pts.append([pid, t, side, 0, stage])
                            label_list.append(0)
                    else:
                        file_list = pid_side_tid_file_dict[pid][side][str(t)]
                        if file_list[0] > 9:
                            break
                        pts.append([pid, t, side, 1, stage])
                        label_list.append(1)
                if len(pts):
                    pid_tid_side.append(pts)

        self.phase = phase
        self.pid_tid_side = pid_tid_side
        self.pid_side_tid_file_dict = pid_side_tid_file_dict

        feat_dict = torch.load('../file/feat.ckpt')

        group_value_dict = dict()
        pid_group_dict = dict()
        for pid in feat_dict:
            if pid not in self.pid_demo_dict:
                continue
            # print(len(feat_dict[pid]))
            for tid,v in feat_dict[pid].items():
                age = float(self.pid_demo_dict[pid]['enroll_age']) + float(tid) / 2
                age = int(max(60, min(80, age)) / 10)

                grp = '{:s}-{:s}-{:s}'.format( self.pid_demo_dict[pid]['sex'], self.pid_demo_dict[pid]['smk_6m'], self.pid_demo_dict[pid]['diab_hist'])
                # grp = '{:s}-{:s}-{:s}'.format( self.pid_demo_dict[pid]['sex'], self.pid_demo_dict[pid]['smk_6m'], str(age))
                if pid not in pid_group_dict:
                    pid_group_dict[pid] = dict()
                pid_group_dict[pid][tid] = grp
                if grp not in group_value_dict:
                    group_value_dict[grp] = []
                group_value_dict[grp].append(v)
        for grp in group_value_dict:
            v = np.array(group_value_dict[grp]).mean(0)
            group_value_dict[grp] = v
        self.group_value_dict = group_value_dict
        self.pid_group_dict = pid_group_dict 


        label_list = [l for l in label_list if l >= 0]
        print('Positive rate: {:1.3f}/{:d}'.format(np.mean(label_list), len(label_list)))

    def stage_vector(self, tl):
        assert len(tl) > 0
        # print(tl)
        v = [tl[0][1]]
        for i in range(1, len(tl)):
            gap = tl[i][0] - tl[i-1][0]
            assert gap >= 1
            for _ in range(gap - 1):
                v.append(tl[i-1][1])
            v.append(tl[i][1])
        ns = 12
        while len(v) < ns:
            v = [0]  + v
        v = v[- ns:]
        assert max(v) <= 9
        return v


        for i in range(1, len(tl)):
            if tl[i][1] < tl[i-1][1]:
                tl[i][1] = tl[i-1][1]

        t_list = [t[0] for t in tl]
        s_list = [t[1] for t in tl]
        t_end = dict()
        for t,s in zip(t_list, s_list):
            t_end[s] = t
        stage_set = set([t[1] for t in tl])



        p = np.zeros((10, 10))
        for i in range(10):
            # v.append(s_cnt.get(i, 0))
            if i in stage_set:
                ti = t_list[s_list.index(i)]
                tj = t_end[i]
                # v.append(tj - ti)
                p[i, i] = tj - ti
                for ii in range(i + 1):
                    for jj in range(i, 10):
                        p[ii, jj] = max(p[ii, jj], p[i,i])
            for j in range(i+1, 10):
                if i in stage_set and j in stage_set:
                    ti = t_list[s_list.index(i)]
                    tj = t_list[s_list.index(j)]
                    # v.append(tj - ti)
                    p[i, j] = tj - ti
                    for ii in range(i + 1):
                        for jj in range(j, 10):
                            p[ii, jj] = max(p[ii, jj], p[i,j])
        for i in range(10):
            for j in range(i, 10):
                v.append(p[i,j])
        # print(len(v))
        # print(len(v), v)
        return v

    def __getitem__(self, index):
        pts = self.pid_tid_side[index]
        assert len(pts) > 0
        n_img = 8
        img_list = []
        labels = []
        vectors = []
        if len(pts) > n_img:
            pts = pts[ - n_img:]
        tl = []
        pid_tid_list = [pts[0][0]]
        feat = []
        for pid, tid, side, label, stage in pts:
            try:
                grp = self.pid_group_dict[pid][tid]
            except:
                age = float(self.pid_demo_dict[pid]['enroll_age']) + float(tid) / 2
                age = int(max(50, min(80, age)) / 10)
                grp = '{:s}-{:s}-{:s}'.format( self.pid_demo_dict[pid]['sex'], self.pid_demo_dict[pid]['smk_6m'], self.pid_demo_dict[pid]['diab_hist'])
                # grp = '{:s}-{:s}-{:s}'.format( self.pid_demo_dict[pid]['sex'], self.pid_demo_dict[pid]['smk_6m'], str(age))
            v = self.group_value_dict[grp]
            feat.append(v)
            img_list.append(self.read_image(pid, tid, side))
            # img_list.append(torch.from_numpy(np.zeros([3, 256,256])))
            # labels.append(label)
            try:
                enrollage = self.pid_demo_dict[pid]['enroll_age']
                age = float(enrollage) + int(tid) / 2.0
            except:
                age = 50
            if age <= 75:
                # labels.append(label - 100)
                labels.append(label)
            else:
                labels.append(label)
            tl.append([int(tid), int(stage)])
            vectors.append(self.stage_vector(tl))
            pid_tid_list.append(tid)




        if len(img_list) < n_img:
            img = torch.from_numpy(np.zeros((3, 256, 256), dtype=np.float32))
            for _ in range(n_img - len(pts)):
                img_list.append(img)
                labels.append(-1)
                vectors.append(vectors[-1])
                pid_tid_list.append('-1')
                feat.append(feat[-1])
        labels_all = np.array(labels, dtype=np.float32)
        labels_all[labels_all < -50] += 100

        feat = np.array(feat, dtype=np.float32)
        return torch.stack(img_list), np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.float32), feat, pid_tid_list

    def read_image(self, pid, tid, side):
        image_name = self.pid_side_tid_file_dict[pid][side][tid][1:]
        np.random.shuffle(image_name)
        image_name = image_name[0]
        # print(image_name)
        image = Image.open(image_name).convert('RGB')
        h,w = image.size
        h = int(h/2)
        w = int(w/2)
        image = image.resize((h,w))
        image = transform(image)
        return image

    def __len__(self):
        return len(self.pid_tid_side) 
