import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from utils import *
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc

def test_detect(data_loader, net, get_pbb, save_dir, config,n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    use_cuda = (n_gpu is not None and n_gpu > 0 and torch.cuda.is_available())
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        # 同時に処理する分割チャンク数（CPUでもバッチ化で高速化の余地あり）
        n_per_run = int(config.get('chunks_per_run', 1))
        if n_per_run < 1:
            n_per_run = 1
        print(data.size())
        splitlist = list(range(0, len(data)+1, n_per_run))
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = data[splitlist[i]:splitlist[i+1]]
            inputcoord = coord[splitlist[i]:splitlist[i+1]]
            if use_cuda:
                input = input.cuda()
                inputcoord = inputcoord.cuda()
            with torch.no_grad():
                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.detach().cpu().numpy())
                else:
                    output = net(input, inputcoord)
            outputlist.append(output.detach().cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature, nzhw=nzhw)[...,0]

        thresh = config.get('pbb_thresh', -3)
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,shortname])
        e = time.time()
        
        np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print
