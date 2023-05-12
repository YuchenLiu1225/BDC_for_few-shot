from data_loader import *
from FSLTask import Baseline
from RelationMap import *
import numpy as np
import torch
from model import FSLRIM
from torch.utils.data import DataLoader
from Utils import *
import time
from torch.optim import lr_scheduler
from torchvision import models

randomseed = 1
torch.cuda.set_device(3)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
args = {
    'dataset': 'derm',
    'root': "../../Derm/derm_bases",
    'n_shot':  1,
    'n_ways':  5,
    'n_queries':  5,#训练时每类抽取作为查询集的图片数，5way，n_queries = 1时查询集共5张图片
    'n_runs':  100, #测试轮次
    'n_gaus': 1000, #采样个数
    'edge_dim': 128,
    'batchSize': 32,
    'lr': 0.0006,
    'seed': 1,
    'optimizer': 'Adamax',
    'lr_decay': 0.9,
    'weight_decay': 0.0003,
    'epoch': 300,
    'lamada1': 0.0005,
    'lamada2': 0.0004,
    'out_dir':"./outfiles/",

    'valruns': 40,
    'test_check_point1': "./outfiles/best_model.nnet.pth",
    'test_check_point2': "./outfiles3/best_model.nnet.pth",
    'test_check_point3': "./outfiles4/best_model.nnet.pth",
    'test_check_point4': "./outfiles5/best_model.nnet.pth",



    #调试及记录
    'feature_dim': 2048,
    'base_num': 23,
    'workers': 1,

}


if __name__ == '__main__':
    for i in range(10):
        print(torch.FloatTensor(1).normal_(1, 1))
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
    np.random.seed(randomseed)
    base_nodes = test.GetBaseFeature()
    model = RIM(feature_dim = args['feature_dim'], base_num = args['base_num'],
            edge_dim = 128, base_nodes = base_nodes, g_dim = args['feature_dim'], n_gaus = 1000, out_dim = out_dim).cuda()

    
    print("Check1\n")
    load_checkpoint=args['test_check_point1']
    state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    time_start = time.time()
    test_acc, graphs =  test_FSL(model, test, args, n_runs = args['n_runs'])
    #filename = "./outfiles/1.txt"
    #with open(filename,'a') as file_object:
    #    file_object.write(json.dumps(args))
    time_end = time.time()
    time_cost = time_end - time_start
    print(" Time:%.3f" % (time_cost))


    print("Check2\n")
    load_checkpoint=args['test_check_point2']
    state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    time_start = time.time()
    test_acc, graphs =  test_FSL(model, test, args, n_runs = args['n_runs'])
    time_end = time.time()
    time_cost = time_end - time_start
    print(" Time:%.3f" % (time_cost))

    print("Check3\n")
    load_checkpoint=args['test_check_point3']
    state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    time_start = time.time()
    test_acc, graphs =  test_FSL(model, test, args, n_runs = args['n_runs'])
    time_end = time.time()
    time_cost = time_end - time_start
    print(" Time:%.3f" % (time_cost))

    print("Check4\n")
    load_checkpoint=args['test_check_point4']
    state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    time_start = time.time()
    test_acc, graphs =  test_FSL(model, test, args, n_runs = args['n_runs'])
    time_end = time.time()
    time_cost = time_end - time_start
    print(" Time:%.3f" % (time_cost))



