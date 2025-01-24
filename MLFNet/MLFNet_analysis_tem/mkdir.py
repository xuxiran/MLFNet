import argparse

import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="pkutmp",choices=["KUL","PKU","DTU","pkutmp"])

parser.add_argument('--model', type=str, default="Notem",choices=
["CNN","STANet","XANet","DCNN","DenseNet","DARNet","DBPNet","ConcatNet",
 "BASE","RSC","Nocat","Nofre","Notem","Dft"])

args, unknown = parser.parse_known_args()
dataset_name = args.dataset
model_name = args.model
savedir = './model/' + dataset_name + '/' + model_name + '/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

savedir = './subresults/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

res_dir = './model_result/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)