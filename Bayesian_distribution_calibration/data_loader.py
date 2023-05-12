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



#random seed
np.random.seed(1) 
random.seed(1)

#读取数据集并提取特征
class DermDataset():
    def __init__(self, dataroot, base_num = 10, cudaid = 'cuda:0'):
        self.cudaid = cudaid
        self.x = Dermdatas(dataroot + "/derm")
        self.base = Dermdatas(dataroot +"/base")
        self.base_num = base_num
        #####处理成数据集形式#####

        #最小类长度
        self._minlen = 1000000

        #backbone
        #self.model = models.resnet101(pretrained = False)
        self.model = models.resnet18(pretrained = True)
        #load_checkpoint="./resnet101-63fe2227.pth"
        #state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
        #self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = torch.device(self.cudaid if torch.cuda.is_available() else "cpu")
        self.features=list(self.model.children())[:-1]#去掉池化层及全连接层
        self.modelout=nn.Sequential(*self.features).to(self.device)


        temp = dict()
        for(img, label) in self.x:
            img = self.modelout(img.unsqueeze(0).to(self.device,torch.float))
            img = img.cpu().detach().numpy().reshape(-1)
            #print(img.shape)
            if label in temp:
                temp[label].append(img)
            else:
                temp[label] = [img]
        self.Datas = []
        for classes in temp.keys():
            classdata = temp[list(temp.keys())[classes]]
            #print(len(classdata))
            #print(classdata[0].shape)
            if len(classdata) < self._minlen:
                self._minlen = len(classdata)
            self.Datas.append(np.array(classdata))
        random.shuffle(self.Datas)
        temp = dict()
        for(img, label) in self.base:
            img = self.modelout(img.unsqueeze(0).to(self.device,torch.float))
            img = img.cpu().detach().numpy().reshape(-1)
            #print(img.shape)
            if label in temp:
                #print(label)
                temp[label].append(img)
            else:
                temp[label] = [img]
        self.base_classes = []
        for classes in temp.keys():
            #print(classes)
            classdata = temp[list(temp.keys())[classes]]
            #print(len(classdata))
            #print(classdata[0].shape)
            if len(classdata) < self._minlen:
                self._minlen = len(classdata)
            self.base_classes.append(np.array(classdata))
        
        #self.Datas: (list类（list类内总数(特征维度2048)))
        #print(self.Datas[0].shape)
        #self.base_classes = self.Datas[-self.base_num:]
        #cut1 = int(0.8 * )
        alllen = len(self.Datas)
        self.x_train = self.Datas[:int(alllen * 0.7)]
        self.x_val = self.Datas[int(alllen * 0.7):int(alllen * 0.85)]
        self.x_test = self.Datas[int(alllen * 0.85):]
        #print(self.x_train[0].shape)

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





    def test(self):
        print("test_successfully!")
        return self.x_train[0][0]
    
    def GenerateRun(self, iRun, cfg, mode = "Test"):
        if mode == "Test":
            x_data = self.x_test
        if mode == "Val":
            x_data = self.x_val
        classes = np.random.permutation(np.arange(len(x_data)))[:cfg["ways"]]
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], x_data[0][0].shape[0]))
        #print(dataset.shape)
        for i in range(cfg['ways']):
            shuffle_indices = np.arange(len(x_data[classes[i]]))
            shuffle_indices = np.random.permutation(shuffle_indices)
            dataset[i] = torch.Tensor(x_data[classes[i]][shuffle_indices])[:cfg['shot']+cfg['queries']]
        return dataset
    
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

        





#输入：数据路径处理成tensor和标签
class Dermdatas(data.Dataset):
    def __init__(self, dataroot):
        self.root = dataroot
        #all_items:(文件名，文件夹，文件具体路径)
        self.all_items=find_classes(os.path.join(self.root))
        #文件夹对应的序号（y）
        self.idx_classes=index_classes(self.all_items)
        #for item in self.all_items:
        #    print(self.idx_classes[item[1]])
    def __getitem__(self, index):
        filename=self.all_items[index][0]
        img=str.join('/',[self.all_items[index][2],filename])

        target=self.idx_classes[self.all_items[index][1]]
        img = Image.open(img).convert('RGB')
        img = img.resize((512, 512))
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        #tensor[3,512,512], label
        return img, target

    def getitem(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return len(self.all_items)




def find_classes(root_dir):
    retour=[]
    #print(root_dir)
    for (root,dirs,files) in os.walk(root_dir):
        #print(dirs)
        for f in files:
            if (f.endswith("jpg")):
                r=root.split('/')
                lr=len(r)
                retour.append((f,r[lr-2]+"/"+r[lr-1],root))
                #print((f,r[lr-2]+"/"+r[lr-1],root))
    print("== Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        #print(i)
        if (not i[1] in idx):
            #print(i[1])
            idx[i[1]]=len(idx)
    for i in idx:
        print(i)
    print("== Found %d classes"% len(idx))
    return idx




#读取数据集并提取特征_old
class DermDataset_old():
    def __init__(self, dataroot, base_num = 10, cudaid = 'cuda:0'):
        self.cudaid = cudaid
        self.x = Dermdatas(dataroot)
        self.base_num = base_num
        #####处理成数据集形式#####

        #最小类长度
        self._minlen = 1000000

        #backbone
        #self.model = models.resnet101(pretrained = False)
        self.model = models.resnet18(pretrained = True)
        #load_checkpoint="./resnet101-63fe2227.pth"
        #state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
        #self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = torch.device(self.cudaid if torch.cuda.is_available() else "cpu")
        self.features=list(self.model.children())[:-1]#去掉池化层及全连接层
        self.modelout=nn.Sequential(*self.features).to(self.device)


        temp = dict()
        for(img, label) in self.x:
            img = self.modelout(img.unsqueeze(0).to(self.device,torch.float))
            img = img.cpu().detach().numpy().reshape(-1)
            #print(img.shape)
            if label in temp:
                temp[label].append(img)
            else:
                temp[label] = [img]
        self.Datas = []
        for classes in temp.keys():
            classdata = temp[list(temp.keys())[classes]]
            #print(len(classdata))
            #print(classdata[0].shape)
            if len(classdata) < self._minlen:
                self._minlen = len(classdata)
            self.Datas.append(np.array(classdata))
        
        #self.Datas: (list类（list类内总数(特征维度2048)）)
        #print(self.Datas[0].shape)
        self.base_classes = self.Datas[-self.base_num:]
        #cut1 = int(0.8 * )
        alllen = len(self.Datas) - self.base_num
        self.x_train = self.Datas[:int(alllen * 0.7)]
        self.x_val = self.Datas[int(alllen * 0.7):int(alllen * 0.85)]
        self.x_test = self.Datas[int(alllen * 0.85): -self.base_num]
        #print(self.x_train[0].shape)

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





    def test(self):
        print("test_successfully!")
        return self.x_train[0][0]
    
    def GenerateRun(self, iRun, cfg, mode = "Test"):
        if mode == "Test":
            x_data = self.x_test
        if mode == "Val":
            x_data = self.x_val
        classes = np.random.permutation(np.arange(len(x_data)))[:cfg["ways"]]
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], x_data[0][0].shape[0]))
        #print(dataset.shape)
        for i in range(cfg['ways']):
            shuffle_indices = np.arange(len(x_data[classes[i]]))
            shuffle_indices = np.random.permutation(shuffle_indices)
            dataset[i] = torch.Tensor(x_data[classes[i]][shuffle_indices])[:cfg['shot']+cfg['queries']]
        return dataset
    
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
        


