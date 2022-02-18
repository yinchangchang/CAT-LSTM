#!/usr/bin/env python
# coding=utf-8

import os
import sys
from tqdm import tqdm
sys.path.append('../tools')
import parse, py_op
import numpy as np
from PIL import Image

import rawpy
import imageio

args = parse.args

type_count_dict = dict()

def img2np(im):
    return np.asarray(im)
def np2img(data):
    return Image.fromarray(np.uint8(data))

def new_wh(im):
    w, h = im.size
    if max(w, h) > 512:
        t = max(w, h) / 512.0
        w = int(w / t)
        h = int(h / t)
        im = im.resize((w,h))
    return im


def low_reso(fi):
    if ' ' in fi:
        return
    if fi.split('.')[-1] not in ['jpg', 'tif', 'dng']:
        return

    new_fi = fi[:-4] + '.png'

    if fi.endswith('.dng'):
        with rawpy.imread(fi) as raw:
            rgb = raw.postprocess()
        im = np2img(rgb)
        im = new_wh(im)
        im.save(new_fi)
    else:
        im = Image.open(fi)
        im = new_wh(im)
        im.save(new_fi)
    cmd = 'rm ' + fi.replace(' ', '\ ')
    os.system(cmd)

     

def get_files(fo):
    fi_list, fo_list = [], []
    for fi in os.listdir(fo):
        if os.path.isdir(os.path.join(fo, fi)):
            fo_list.append(os.path.join(fo, fi))
        else:
            fi_list.append(os.path.join(fo, fi))
            t = fi.split('.')[-1]
            if t not in type_count_dict:
                type_count_dict[t] = 1
            else:
                type_count_dict[t] += 1
    for sub_fo in fo_list:
        sub_fi = get_files(sub_fo)
        fi_list += sub_fi
    return fi_list

def unzip_files(fi):
    if not fi.endswith('.ncbi_enc'):
        return
    path = os.path.dirname(fi)
    fi_name = fi.split('/')[-1]
    tar_name = fi_name.replace('.ncbi_enc', '')
    cmd = '''
        cd {:s}
        vdb-decrypt --ngc ~/prj_26125_D31459.ngc {:s}
        tar xvf {:s}
        rm {:s}
        '''.format(path, fi_name, tar_name, tar_name)
    print(cmd)
    os.system(cmd)
    # print(err)
    for img in tqdm(os.listdir(path)):
        try:
            low_reso(os.path.join(path, img))
        except:
            pass

def my_mkdir(path):
    p_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(p_path):
        my_mkdir(p_path)
    os.mkdir(path)

def my_cp(fi, new_fi):
    path = '/'.join(new_fi.split('/')[:-1])
    print(new_fi, path)
    if not os.path.exists(path):
        my_mkdir(path)
    cmd = 'cp {:s} {:s}'.format(fi, new_fi)
    print(cmd)
    os.system(cmd)

def main():
    fi_list = get_files(os.path.join(args.src_dir, 'AREDS'))
    for fi in tqdm(fi_list):
        new_fi = fi.replace(args.src_dir, args.data_dir)
        my_cp(fi, new_fi)
        unzip_files(new_fi)
        # os.system('rm ' + fi)
        # break

if __name__ == '__main__':
    main()
