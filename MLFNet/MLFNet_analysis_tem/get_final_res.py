import numpy as np
import torch
import argparse

import os
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="KUL",choices=["KUL","PKU","DTU"])
parser.add_argument('--model', type=str, default="ConcatNet")
parser.add_argument('--tems', type=int, default=0)
parser.add_argument('--teme', type=int, default=500)

args, unknown = parser.parse_known_args()

dataset_name = args.dataset
model_name = args.model
# get all the res.csv



res = torch.zeros(4,1)

savedir = './subresults/'
for sbfold in range(4):
    csvname = savedir + dataset_name + '_' + model_name + '_' + str(sbfold)+ '_' + str(args.tems) + '_' + str(args.teme) + '.csv'
    res_sbfold = np.loadtxt(csvname,delimiter=',')
    res[sbfold] = torch.tensor(res_sbfold[sbfold])

# save the result with the name of dataset and model_name
res_dir = './model_result/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
np.savetxt('./model_result/' + dataset_name+'_'+model_name+ '_' + str(args.tems) + '_' + str(args.teme) + '.csv', res.numpy(), delimiter=',')


