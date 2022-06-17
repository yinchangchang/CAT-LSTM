# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='medical caption GAN')

parser.add_argument( '--data-dir', type=str, default='/home/yin/data/dbgap/EDO', help='data files directory')
parser.add_argument( '--file-dir', type=str, default='/home/yin/code/dbgap/file', help='data files directory')
# parser.add_argument( '--src-dir', type=str, default='/media/yin/Elements/data/dbgap/84157', help='data files directory')
# parser.add_argument( '--src-dir', type=str, default='/media/yin/A67A6AF37A6ABFA3/dataset/dbgap/84155', help='data files directory')
parser.add_argument( '--src-dir', type=str, default='/media/yin/A67A6AF37A6ABFA3/dataset/dbgap/84298', help='data files directory')

parser.add_argument( '--lr', type=float, default=0.0001)
parser.add_argument( '--bs', type=int, default=8)
parser.add_argument( '--mode', type=str, default='detection')
parser.add_argument( '--phase', type=str, default='train')
parser.add_argument( '--hard-mining', type=int, default=0)


args = parser.parse_args()
