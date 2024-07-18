import os
import time
import json
import pickle
import itertools
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn
from torch.optim import LBFGS
from torch.distributions.multivariate_normal import MultivariateNormal

#from FSLTask import FSLTaskMaker
#from utils.io_utils import DataWriter, logger

from data_loader import *
from BayesianInference import *
import numpy as np
#from model import FSLRIM
from torch.utils.data import DataLoader
from Utils import *
import time
from torch.optim import lr_scheduler
from torchvision import models
import json


randomseed = 0
torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    args = {
    #few-shot configs
    'n_shot':  1, #n-shot
    'n_ways':  5, #n-way
    'n_queries':  5,#size of query set
    'n_runs':  100, #episodes during test phase
    'valruns': 40, #episodes during validation phase

    #parameters for Bayesian relation inference module
    'edge_dim': 128, #feature dimension for edge-embedding
    'batchSize': 256, #batch size
    'lamada1': 0.005, #lambda1 for kl-loss calculation
    'lamada2': 0.004, #lambda2 for kl-loss calculation
    'n_gaus': 1000, #number of Gaussian graph samplings
    'base_num': 10, #number of base classes

    #model hyperparameter
    'lr': 0.0002, #learning rate
    'seed': 1, #random seed
    'optimizer': 'Adam', #optimizer
    'lr_decay': 0.9, #learning decay for optimizer
    'weight_decay': 0.0003, #weight decay for optimizer
    'epoch': 50, #training epochs 
    
    'out_dir':"./outfile/", # model parameter save path
    'TV_lr': 0.0001, # learning rate during validation & test phase
    'workers': 1,

    }

    #loading dataset
    train_path = "mini_base_features.plk" #your path of .plk file
    val_path = "min_val_features.plk"
    test_path = "mini_novel_features.plk"
    test = DermDataset(train_path = train_path, val_path = val_path, test_path = test_path, base_num = args['base_num'], cudaid = device)
    tr_X, tr_Y, out_dim = test.Gen_DataLoader()
    args['feature_dim'] = tr_X[0].shape[0] #feature_dim after passing through backbone model
    trDataset = SimpleDataset(tr_X, tr_Y)
    trGen = DataLoader(trDataset,
                       batch_size=args['batchSize'],
                       shuffle=True,
                       num_workers=args['workers'])
    

    #set random seed
    np.random.seed(randomseed)
    #initialize base nodes
    base_nodes = test.GetBaseFeature()

    #The proposed model
    model = RIM(feature_dim = args['feature_dim'], base_num = args['base_num'],
            edge_dim = args['edge_dim'], base_nodes = base_nodes, g_dim = args['feature_dim'], n_gaus = 1000, out_dim = out_dim).cuda()

    
    #initialze optimizer
    if args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])  #,momentum=0.5,nesterov=True)
    elif args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adam+AMS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], amsgrad=True)
    elif args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], amsgrad=False)
    elif args['optimizer'] == 'AdamW+AMS':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], amsgrad=True)
    elif args['optimizer'] == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    if args['lr_decay']>0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args['lr_decay'])
    
    #We use Adam optimizer during validation & test
    op2 = torch.optim.Adam(model.parameters(), lr=args['TV_lr'], weight_decay=args['weight_decay'])
    #loss function
    loss_func = one_hot_CrossEntropy()

    # Train, Valid & test
    val_loss_before = np.inf
    best_acc = 0
    best_acc2 = 0
    count_epoch = 0

    for epoch in range(args['epoch']):
        time_start = time.time()
        tr_acc, tr_loss, tr_kl1, tr_kl2 = train_epoch(model, trGen, args, loss_func, optimizer, epoch)
        
        #if training accuarcy > 0.8, we can validate and test the model
        if tr_acc > 0.8:
            torch.save(model.state_dict(), args['out_dir'] + 'best_model.nnet.pth')
            val_acc = val_FSL(model, test, args, epoch, loss_func, op2, n_runs = args['valruns'])
            test_acc =  test_FSL(model, test, args, epoch, loss_func, op2, n_runs = args['n_runs'])

            if val_acc > best_acc:
                best_acc = val_acc
            if test_acc > best_acc2 and test_acc > best_acc:
                best_acc2 = test_acc
                torch.save(model.state_dict(), args['out_dir'] + 'test_best_model' +str(test_acc) + '.nnet.pth')
                print(" U ", end='')

        if args['lr_decay']>0:
            lr_epoch = scheduler.get_last_lr()
        else:
            lr_epoch = learning['lr']
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost), 'lr:', lr_epoch)



if __name__ == '__main__':
    main()
