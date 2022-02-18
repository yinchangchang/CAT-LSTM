# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='medical caption GAN')

parser.add_argument( '--data-dir', type=str, default='~/data/dbgap/EDO', help='data files directory')
parser.add_argument( '--file-dir', type=str, default='~/code/dbgap/file', help='data files directory')
parser.add_argument( '--src-dir', type=str, default='~/dbgap/', help='data files directory')

parser.add_argument( '--lr', type=float, default=0.0001)
parser.add_argument( '--use-cl', type=int, default=1, help='Use Contrastive Learning')
parser.add_argument( '--bs', type=int, default=4)
parser.add_argument( '--mode', type=str, default='detection')
parser.add_argument( '--phase', type=str, default='train')
parser.add_argument( '--hard-mining', type=int, default=0)


args = parser.parse_args()
