#!/usr/bin/env python
# coding=utf-8

import os
import sys
from multiprocessing import Pool
from tqdm import tqdm
sys.path.append('../tools')
import parse, py_op
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import rawpy
import imageio

args = parse.args


def id_mapping():
    id_mapping_dict = dict()
    fi = '../file/AREDS_Image_Mapping_ImageDir_ID2.txt'
    for line in open(fi):
        id1, id2 = line.strip().split()
        try:
            _ = int(id1)
            id_mapping_dict[id1] = id2
            id_mapping_dict[id2] = id1
        except:
            print(line)
    py_op.mywritejson(os.path.join(args.file_dir, 'id_mapping_dict.json'), id_mapping_dict)

def generate_label_file():
    pid_tid_stage_dict = dict()
    fi_list = [os.path.join(args.data_dir, '../GRU/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c2.GRU/PhenotypeFiles/phs000001.v3.pht000375.v2.p1.c2.fundus.GRU.txt'), 
            os.path.join(args.data_dir, '../EDO/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c1.EDO/PhenotypeFiles/phs000001.v3.pht000375.v2.p1.c1.fundus.EDO.txt')]
    for fi in fi_list:
        head = []
        for line in open(fi):
            if line.startswith('dbGaP'):
                head = line.strip().split('\t')

            elif len(head):
                if len(line.split('\t')) == 50:
                    data = line.split('\t')
                    pid = data[head.index('ID2')]
                    tid = int(data[head.index('VISNO')])
                    for side in ['LE', 'RE']:
                        key = 'AMDSEV' + side
                        try:
                            stage = int(data[head.index(key)])
                        except:
                            print('empty stage')
                            pass

                        if pid not in pid_tid_stage_dict:
                            pid_tid_stage_dict[pid] = { 'LE': { }, 'RE': { } }
                        pid_tid_stage_dict[pid][side][tid] = stage
    py_op.mywritejson(os.path.join(args.file_dir, 'pid_tid_stage_dict.json'), pid_tid_stage_dict)

def get_files(fo):
    fi_list, fo_list = [], []
    for fi in os.listdir(fo):
        if os.path.isdir(os.path.join(fo, fi)):
            fo_list.append(os.path.join(fo, fi))
        else:
            fi_list.append(os.path.join(fo, fi))
    for sub_fo in fo_list:
        sub_fi = get_files(sub_fo)
        fi_list += sub_fi
    return fi_list

def main():
    id_mapping()
    generate_label_file()
    pid_tid_stage_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_tid_stage_dict.json'))
    id_mapping_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_mapping_dict.json'))
    fi_list = get_files(os.path.join(args.data_dir, 'AREDS'))


    pid_side_tid_file_dict = dict()
    cnt_img = 0

    for fi in fi_list:
        if 'ref' in fi.lower():
            continue
        pid, tid = fi.split('/')[-1].split('_')[:2]
        try:
            tid = str(int(tid))
        except:
            continue
        assert 'LE' in fi.split('/')[-1].upper() or 'RE' in fi.split('/')[-1].upper()
        if 'LE' in fi.split('/')[-1].upper():
            side = 'LE'
        else:
            side = 'RE'

        if len(pid) >= 5 and pid in id_mapping_dict:
            pid = id_mapping_dict[pid]

        if pid in pid_tid_stage_dict:
            if pid in pid_tid_stage_dict:
                if pid not in pid_side_tid_file_dict:
                    pid_side_tid_file_dict[pid] = { 'LE': { }, 'RE': { } }
                if tid not in pid_tid_stage_dict[pid][side]:
                    continue
                if tid not in pid_side_tid_file_dict[pid][side]:
                    pid_side_tid_file_dict[pid][side][tid] = [pid_tid_stage_dict[pid][side][tid]]
                pid_side_tid_file_dict[pid][side][tid].append(fi)
                cnt_img += 1
    print(cnt_img)
    print(len(pid_side_tid_file_dict))

    pid_list = list(pid_side_tid_file_dict.keys())
    np.random.shuffle(pid_list)
    n_train = int(0.7 * len(pid_list))
    pid_side_tid_file_dict['train'] = pid_list[:n_train]
    pid_side_tid_file_dict['valid'] = pid_list[n_train:]
    py_op.mywritejson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'), pid_side_tid_file_dict)

def renew_dataset():
    pid_side_tid_file_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'))
    pid_demo_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_demo_dict.json'))
    if 'train' in pid_side_tid_file_dict:
        pid_side_tid_file_dict.pop('train')
        pid_side_tid_file_dict.pop('valid')
    cnt = 0
    all = 0
    side_tid_list = []
    for pid, std in pid_side_tid_file_dict.items():
        if type(std) is dict:
            side_list = ['LE', 'RE']
            for i in range(2):
                side = side_list[i]
                oside = side_list[1 - i]
                for t in std[side]:
                    if t not in std[oside]:
                        cnt += 1
                        side_tid_list.append([pid, side, t])

                    all += 1
    print(cnt, all)
    for pid, side, tid in side_tid_list:
        pid_side_tid_file_dict[pid][side].pop(tid)
    for pid in list(pid_side_tid_file_dict.keys()):
        d = pid_side_tid_file_dict[pid]['RE']
        if len(d) < 6:
            pid_side_tid_file_dict.pop(pid)
    pid_list = list(pid_side_tid_file_dict.keys())
    np.random.shuffle(pid_list)
    n_train = int(0.7 * len(pid_list))
    pid_side_tid_file_dict['train'] = sorted(pid_list[:n_train])
    pid_side_tid_file_dict['valid'] = sorted(pid_list[n_train:])
    py_op.mywritejson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'), pid_side_tid_file_dict)


def delete_file(f):
    try:
        image = Image.open(f).convert('RGB')
    except:
        cmd ='rm ' + f
        print(cmd)
        os.system(cmd)

def delete():
    pid_side_tid_file_dict = py_op.myreadjson(os.path.join(args.file_dir, 'pid_side_tid_file_dict.json'))
    cnt = 0
    side_tid_list = []
    p = Pool(64)
    for pid, std in tqdm(pid_side_tid_file_dict.items()):
        if type(std) is dict:
            side_list = ['LE', 'RE']
            for i in range(2):
                side = side_list[i]
                for t in std[side]:
                    assert len(std[side][t]) > 1
                    for f in std[side][t][1:]:
                        p.apply_async(delete_file, args=(f,))
    p.close()
    p.join()



if __name__ == '__main__':
    main()
    renew_dataset()
