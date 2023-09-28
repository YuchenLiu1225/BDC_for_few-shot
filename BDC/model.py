from RelationMap3 import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
from torchvision import models
from data_loader import *

_randStates = None
_rsCfg = None
_maxRuns = 10000
_min_examples = -1
data = None
labels = None
dsName = None

class FSLRIMmodel(nn.Module):
    def __init__(self, n_shot, n_ways, n_queries, feature_dim, base_num,
     edge_dim, base_nodes, g_dim = 128, dropout = 0.1, n_gaus = 1000):
        super(FSLRIMmodel, self).__init__()
        self.n_shot = n_shot
        self.n_ways = n_ways
        self.n_queries = n_queries
        self.g_dim = g_dim
        self.n_lsamples = n_ways * n_shot
        self.n_usamples = n_ways * n_queries
        self.n_samples = self.n_lsamples + self.n_usamples
        self.RIM = RIM(feature_dim = feature_dim, base_num = base_num,
            edge_dim = edge_dim, base_nodes = base_nodes, g_dim = self.g_dim, n_gaus = n_gaus / self.n_shot)
        #self.classifier = models.resnet18(pretrained = False)
        self.classifier = LogisticRegression(max_iter=1000)
    def forward(self, support_data, support_label, query_data, query_label):
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/self.n_shot)                                                                        

def cal_correct(softmax_out, y_true):
    y_pred = torch.argmax(softmax_out.data, dim=1)
    correct = ((y_pred == y_true.long()).sum()).cpu().numpy()
    return correct


def FSLRIM(dataroot, feature_dim, base_num, edge_dim,
n_shot = 1, n_ways = 5, n_queries = 1,  g_dim = 128, dropout = 0.1, n_gaus = 1000, device = "cuda:0", n_runs = 100):
    global data
    data = DermDataset(dataroot, base_num = base_num, cudaid = device)
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    base_nodes = data.GetBaseFeature()
    ndatas = GenerateRunSet(end=n_runs, cfg=cfg) #(n_runs, n_samples, 2048)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
    5).clone().view(n_runs, n_samples)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
                    5).clone().view(n_runs, n_samples)
    print(labels.shape)
    print(ndatas.shape)
    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))

    model = RIM(feature_dim = feature_dim, base_num = base_num,
            edge_dim = edge_dim, base_nodes = base_nodes, g_dim = g_dim, n_gaus = int(n_gaus / n_shot)).cuda()
    #model = FSLRIMmodel(n_ways = n_ways, n_shot = n_shot, n_queries = n_queries, feature_dim = feature_dim, base_num = base_num,
    # edge_dim = edge_dim, base_nodes = base_nodes, n_gaus = n_gaus)
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        #print(support_data.shape)
        #print(support_label)
        # ---- Tukey's transform
        beta = 0.5
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(n_gaus/n_shot)

        for j in range(n_lsamples):
            #模型先训练好再放进来采样
            sampled_data.append(model(support_data[j]).cpu().detach().numpy())
            #mean, cov = distribution_calibration(support_data[j], base_means, base_cov, k=2)
            #sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = sampled_data
        Y_aug = sampled_label
        #X_aug = np.concatenate([support_data, sampled_data])
        #Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        #分类器无所谓
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)

        predicts = classifier.predict(query_data)


        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        with open("log.txt", "a+") as f:
            f.write("run %d, ACC: %f\n"%(i, acc))
    print('derm: %d way %d shot  ACC : %f'%(n_ways,n_shot,float(np.mean(acc_list))))
    with open("log.txt", "a+") as f:
        f.write('derm: %d way %d shot  ACC : %f'%(n_ways,n_shot,float(np.mean(acc_list))))



def GenerateRunSet(cfg, start = 0, end = 100):
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 1}
    global data

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], 2048))
    for iRun in range(end-start):
        dataset[iRun] = data.GenerateRun(start+iRun, cfg) #(ways, shot+queries, data.shape)
    #print(dataset.shape)
    return dataset
