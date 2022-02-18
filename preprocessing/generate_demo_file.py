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

def generate_demo_file():

    pid_demo_dict = dict()
    fi_list = [
            os.path.join(args.data_dir, '../GRU/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c2.GRU/PhenotypeFiles/phs000001.v3.pht000373.v2.p1.c2.enrollment_randomization.GRU.txt'), 
            os.path.join(args.data_dir, '../EDO/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c1.EDO/PhenotypeFiles/phs000001.v3.pht000373.v2.p1.c1.enrollment_randomization.EDO.txt'),
            os.path.join(args.data_dir, '../GRU/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c2.GRU/PhenotypeFiles/phs000001.v3.pht002479.v1.p1.c2.sunlight.GRU.txt'), 
            os.path.join(args.data_dir, '../EDO/PhenoGenotypeFiles/RootStudyConsentSet_phs000001.AREDS.v3.p1.c1.EDO/PhenotypeFiles/phs000001.v3.pht002479.v1.p1.c1.sunlight.EDO.txt'),
            ]
    for fi in fi_list:
        head = []
        for line in open(fi):
            if line.startswith('dbGaP'):
                head = line.strip().split('\t')

            elif len(head):
                if len(line.split('\t')) == 103:
                    data = line.strip().split('\t')
                    pid = data[head.index('ID2')]
                    if pid not in pid_demo_dict:
                        pid_demo_dict[pid] = dict()

                    pid_demo_dict[pid]['enroll_age']= data[head.index('ENROLLAGE')]
                    pid_demo_dict[pid]['race'] = data[head.index('RACE')]
                    pid_demo_dict[pid]['sex'] = data[head.index('SEX')]
                    pid_demo_dict[pid]['bmi'] = data[head.index('BMI_R')]

                    pid_demo_dict[pid]['smk_age'] = data[head.index('SMKAGEST')]
                    pid_demo_dict[pid]['smk_packs'] = data[head.index('SMKPACKS')]
                    pid_demo_dict[pid]['smk_current'] = data[head.index('SMKCURR')]
                    pid_demo_dict[pid]['smk_no_cig'] = data[head.index('SMKNOCIG')]
                    pid_demo_dict[pid]['smk_quit_age'] = data[head.index('SMKAGEQT')]
                    pid_demo_dict[pid]['smk_6m'] = data[head.index('SMOKEDYN')]

                    pid_demo_dict[pid]['diab_hist'] = data[head.index('DIABETYN')]
                    pid_demo_dict[pid]['diab_age'] = data[head.index('DIABAGE')]
                    pid_demo_dict[pid]['diab_insulin_years'] = data[head.index('DIABINS')]
                    pid_demo_dict[pid]['diab_pill_years'] = data[head.index('DIABPILL')]
                    pid_demo_dict[pid]['diab_diet_years'] = data[head.index('DIABDIET')]
                if len(line.split('\t')) == 3:
                    data = line.strip().split('\t')
                    pid = data[head.index('ID2')]
                    if pid not in pid_demo_dict:
                        continue
                    pid_demo_dict[pid]['sun_light'] = data[head.index('EFF_YR')]

    py_op.mywritejson(os.path.join(args.file_dir, 'pid_demo_dict.json'), pid_demo_dict)


def main():
    id_mapping()
    generate_demo_file()

if __name__ == '__main__':
    main()
