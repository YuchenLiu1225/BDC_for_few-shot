from data_loader import *
from FSLTask import Baseline
from RelationMap3 import *
import numpy as np
from model import FSLRIM
from torch.utils.data import DataLoader
from Utils import *
import time
from torch.optim import lr_scheduler
from torchvision import models
import json

randomseed = 1
torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args = {
    'dataset': 'derm',
    'root': "../../Derm/derm_bases",#littletest",
    'n_shot':  1,
    'n_ways':  5,
    'n_queries':  5,#训练时每类抽取作为查询集的图片数，5way，n_queries = 1时查询集共5张图片
    'n_runs':  100, #训练轮次
    'n_gaus': 1000, #采样个数
    'edge_dim': 256,
    'batchSize': 64,
    'lr': 0.00015,
    'seed': 1,
    'optimizer': 'Adam',
    'lr_decay': 0.9,
    'weight_decay': 0.0003,
    'epoch': 800,
    'lamada1': 0.0005,
    'lamada2': 0.0004,
    'out_dir':"./outfiles_softmax/",

    'valruns': 40,



    #调试及记录
    'feature_dim': 2048,
    'base_num': 23,
    'workers': 1,

}


if __name__ == '__main__':
    filename = "./outfiles/log.txt"
    with open(filename,'a') as file_object:
        file_object.write(json.dumps(args))

    print(args)
    test = DermDataset(dataroot = args['root'], base_num = args['base_num'], cudaid = device)
    tr_X, tr_Y, out_dim = test.Gen_DataLoader()
    args['feature_dim'] = tr_X[0].shape[0]
    print(tr_X[0].shape)
    #print(tr_Y)
    trDataset = SimpleDataset(tr_X, tr_Y)
    trGen = DataLoader(trDataset,
                       batch_size=args['batchSize'],
                       shuffle=True,
                       num_workers=args['workers'])
    """
    val_X, val_Y = test.Gen_DataLoader("Val")
    valDataset = SimpleDataset(val_X, val_Y)
    valGen = DataLoader(valDataset,
                       batch_size=args['batchSize'],
                       shuffle=False,
                       num_workers=args['workers'])
    """
    np.random.seed(randomseed)
    base_nodes = test.GetBaseFeature()

    model = RIM(feature_dim = args['feature_dim'], base_num = args['base_num'],
            edge_dim = args['edge_dim'], base_nodes = base_nodes, g_dim = args['feature_dim'], n_gaus = 1000, out_dim = out_dim).cuda()
    #model2 = TestModel(feature_dim = args['feature_dim'], base_num = args['base_num'],
    #        edge_dim = 128, base_nodes = base_nodes, g_dim = args['feature_dim'], n_gaus = 1000, out_dim = out_dim).cuda()
    #load_checkpoint="./outfiles_new1/best_model775.nnet.pth"
    #state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    #model.load_state_dict(state_dict)
    
    
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
    
    loss_func = one_hot_CrossEntropy()

    # Train & Valid 1 epoch
    val_loss_before = np.inf
    best_acc = 0
    best_acc2 = 0
    count_epoch = 0

    for epoch in range(args['epoch']):
        time_start = time.time()
        #print("lr:",)
        tr_acc, tr_loss, tr_kl1, tr_kl2 = train_epoch(model, trGen, args, loss_func, optimizer, epoch)
        #tr_acc, tr_loss, tr_kl1, tr_kl2 = train_epoch2(model2, trGen, args, loss_func, optimizer, epoch)

        #val_acc = val_FSL(model, test, args, epoch, n_runs = args['valruns'])
        #test_acc =  test_FSL(model, test, args, n_runs = args['n_runs'])

        if args['lr_decay']>0:
            lr_epoch = scheduler.get_last_lr()
        else:
            lr_epoch = learning['lr']#scheduler.get_last_lr()
        if tr_acc > 0.70:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), args['out_dir'] + 'best_model_' + str(tr_acc) + '.nnet.pth')
        '''
        if val_acc > best_acc:
            best_acc = val_acc
            if test_acc > best_acc2:
                best_acc2 = test_acc
                torch.save(model.state_dict(), args['out_dir'] + 'best_model' + '.nnet.pth')
                print(" U ", end='')
        '''
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost), 'lr:', lr_epoch)

    #test_node = test.test()
   
    #x, kl1, kl2 = model(test_node)
    #print(x.shape)
    #print(kl1)
    #print(kl2)


    #test.__getitem__(1)