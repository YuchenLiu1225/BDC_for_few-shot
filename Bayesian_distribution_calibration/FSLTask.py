from data_loader import *
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch

_randStates = None
_rsCfg = None
_maxRuns = 10000
_min_examples = -1
data = None
labels = None
dsName = None



def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov


def Baseline(dataroot, n_shot = 1, n_ways = 5, n_queries = 1, n_runs = 100):
    global data
    data = DermDataset(dataroot)
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    ndatas = GenerateRunSet(end=n_runs, cfg=cfg) #(n_runs, n_samples, 2048)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
    5).clone().view(n_runs, n_samples)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
                    5).clone().view(n_runs, n_samples)
    print(labels.shape)
    print(ndatas.shape)
    # ---- Base class statisticsopen
    base_means, base_cov = data.GetBasestats()
    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
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
        num_sampled = int(750/n_shot)
        for j in range(n_lsamples):
            mean, cov = distribution_calibration(support_data[j], base_means, base_cov, k=2)
            sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = np.concatenate([support_data, sampled_data])
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
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
    print(dataset.shape)
    return dataset
