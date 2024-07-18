import torch.utils.data as data
import os
import errno
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle



#random seed
np.random.seed(1) 
random.seed(1)

#读取数据集并提取特征
class DermDataset():
    def __init__(self, train_path, val_path, test_path, base_num = 10, cudaid = 'cuda:0'):
        self.cudaid = cudaid
        self.base_num = base_num
        self.x_train = []
        self.x_test = []
        self.x_val = []
        print(f"* Reading Base Features from {train_path}")
        with open(train_path, 'rb') as fp:
            data = pickle.load(fp)
            i = 1
            for key in data.keys():
                feature = np.array(data[key])
                i += 1
                self.x_train.append(feature)

        print(f"* Reading Test Features from {test_path}")
        with open(test_path, 'rb') as fp:
            data = pickle.load(fp)
            i = 1
            for key in data.keys():
                feature = np.array(data[key])
                i += 1
                self.x_test.append(feature)

        print(f"* Reading Val Features from {val_path}")
        with open(val_path, 'rb') as fp:
            data = pickle.load(fp)
            i = 1
            for key in data.keys():
                feature = np.array(data[key])
                i += 1
                self.x_val.append(feature)      
        
        #split base classes from training data
        self.base_classes = self.x_train[:self.base_num]
        self.x_train = self.x_train[self.base_num:]
        print("Dataset has %d baseclasses, %d trainclasses, %d valclasses, %d testclasses"%(len(self.base_classes), len(self.x_train), len(self.x_val), len(self.x_test)))
    

    def Gen_DataLoader(self, mode = "Train"):
        y_list = []
        x_list = []
        all_len = 0
        lens = 0
        if mode == "Train":
            lens = len(self.x_train)
            for i in range(len(self.x_train)):
                x_list.extend(self.x_train[i])
                y = np.full(self.x_train[i].shape[0], i)
                all_len = all_len + self.x_train[i].shape[0]
                y_list.extend(y)
        elif mode == "Val":
            lens = len(self.x_val)
            for i in range(len(self.x_val)):
                x_list.extend(self.x_val[i])
                y = np.full(self.x_val[i].shape[0], i)
                all_len = all_len + self.x_val[i].shape[0]
                y_list.extend(y)
        elif mode == "Test":
            lens = len(self.x_test)
            for i in range(len(self.x_test)):
                x_list.extend(self.x_test[i])
                y = np.full(self.x_test[i].shape[0], i)
                all_len = all_len + self.x_test[i].shape[0]
                y_list.extend(y)
        return np.array(x_list), np.array(y_list), len(self.x_train)
    
    #generate an episode
    def GenerateRun(self, iRun, cfg, mode = "Test"):
        if mode == "Test":
            x_data = self.x_test
        if mode == "Val":
            x_data = self.x_val
        classes = np.random.permutation(np.arange(len(x_data)))[:cfg["ways"]]
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], x_data[0][0].shape[0]))
        for i in range(cfg['ways']):
            shuffle_indices = np.arange(len(x_data[classes[i]]))
            shuffle_indices = np.random.permutation(shuffle_indices)
            dataset[i] = torch.Tensor(x_data[classes[i]][shuffle_indices])[:cfg['shot']+cfg['queries']]
        return dataset
    
    #get mean&cov of base classes (for DC method)
    def GetBasestats(self):
        base_means = []
        base_cov = []
        for base_class in self.base_classes:
            base_feature = np.array(base_class)
            mean = np.mean(base_feature, axis = 0)
            cov = np.cov(base_feature.T)
            base_means.append(mean)
            base_cov.append(cov)
        return base_means, base_cov
    
    #We use the mean of all features in each base class as its feature
    def GetBaseFeature(self):
        base_nodes = np.empty((len(self.base_classes), self.base_classes[0].shape[-1]))#torch.zeros(len(self.base_classes), self.base_classes[0].shape[-1])
        it = 0
        for base_class in self.base_classes:
            x0 = base_class[0]
            for i in range(1, base_class.shape[0]):
                x0 = x0 + base_class[i]
            x0 = x0 / base_class.shape[0]
            base_nodes[it] = x0
            it = it + 1
        return base_nodes
    
    def GenAll(self):
        return self.Datas