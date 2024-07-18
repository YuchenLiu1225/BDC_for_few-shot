import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import Dataset, DataLoader

def one_hot(y, class_count):
    y = y.cpu()
    return torch.eye(class_count)[y, :]


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), \
            'The number of inputs(%d) and targets(%d) does not match.' % (len(x), len(y))
        self.x = x
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        y = one_hot(y, x.shape[-1]).cuda()
        #print(y.shape)
        P_i = torch.nn.functional.softmax(x, dim=1).cuda()
        #print(P_i)
        loss = y*torch.log(P_i + 0.00000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss


def train_epoch(model, train_loader, learning, my_loss, optimizer, epoch, threshold=10):
    model.train()
    acc = 0
    total = 0
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    test_acc_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, _= x.size()

        optimizer.zero_grad()
        out_x, kl_g, kl_b, graph, _ = model(x[0])
        out_x = out_x.unsqueeze(0)

        for i in range(1,b):
            out_x1, kl_g1, kl_b1, _, _ = model(x[i])
            kl_g = kl_g + kl_g1
            kl_b = kl_b + kl_b1
            out_x1 = out_x1.unsqueeze(0)
            out_x = torch.cat((out_x, out_x1), dim = 0)

        loss_kl1 = learning['lamada1'] * kl_g
        loss_kl2 = learning['lamada2'] * kl_b
        loss = my_loss(out_x, y) + loss_kl1 + loss_kl2
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        pred = torch.argmax(out_x.data, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        acc2 = (pred == y).double().mean(dim=(-1)).detach().cpu().numpy().ravel()
        test_acc_list += acc2.tolist()
        total += len(y)
        loss_list.append(loss.item())
        loss_kl1_list.append(loss_kl1.item())
        loss_kl2_list.append(loss_kl2.item())
    print(graph)
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    acc_ci = 1.96 * float(np.std(test_acc_list) / np.sqrt(len(test_acc_list)))
    print("[TR]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f +/- %f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total), acc_ci))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean

class dataset_prediction(Dataset):
    '''
    将传入的数据集，转成Dataset类，方面后续转入Dataloader类
    注意定义时传入的data_features,data_target必须为numpy数组
    '''
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.target = torch.from_numpy(data_target)
        
    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

def train_test(model, datas, labels, learning, my_loss, optimizer):
    model.train()
    acc = 0
    acc2 = 0
    acc3 = -1
    count = 0
    total = 0
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    datas = np.repeat(datas, 10, axis = 0)
    labels = np.repeat(labels, 10, axis = 0)

    
    train_set = dataset_prediction(data_features=datas, data_target=labels)
    train_loader = DataLoader(dataset=train_set,
                                batch_size=labels.shape[0],
                                shuffle=True,
                                drop_last=False)

    while acc2 < 0.85:
        for batch_idx, (x, y) in enumerate(train_loader):
            # If you run this code on CPU, please remove the '.cuda()'
            x = x.cuda()
            y = y.cuda()
            b, _= x.size()

            optimizer.zero_grad()
            out_x, kl_g, kl_b, graph, _ = model(x[0], mode = "TV")
            out_x = out_x.unsqueeze(0)

            for i in range(1,b):
                out_x1, kl_g1, kl_b1, _, _ = model(x[i], mode = "TV")
                kl_g = kl_g + kl_g1
                kl_b = kl_b + kl_b1
                out_x1 = out_x1.unsqueeze(0)
                out_x = torch.cat((out_x, out_x1), dim = 0)

            loss_kl1 = learning['lamada1'] * kl_g
            loss_kl2 = learning['lamada2'] * kl_b
            loss = my_loss(out_x, y) + loss_kl1 + loss_kl2
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            pred = torch.argmax(out_x.data, 1)
            acc += ((pred == y).sum()).cpu().numpy()
            total += len(y)
            loss_list.append(loss.item())
            loss_kl1_list.append(loss_kl1.item())
            loss_kl2_list.append(loss_kl2.item())

        loss_mean = np.mean(loss_list)
        loss_kl1_mean = np.mean(loss_kl1_list)
        loss_kl2_mean = np.mean(loss_kl2_list)

        acc2 = acc / total
        if acc3 == acc2:
            count += 1
            assert count < 50, 'infinate train_test_epoch'
        else:
            acc3 = acc2
    return acc / total

def val_FSL(model, data, learning, epoch, my_loss, optimizer, n_runs = 20):
    cfg = {'shot': learning['n_shot'], 'ways': learning['n_ways'], 'queries': learning['n_queries'], 'feature_dim': learning['feature_dim']}
    n_gaus = learning['n_gaus']
    n_shot = learning['n_shot']
    n_ways = learning['n_ways']
    n_queries = learning['n_queries']
    n_lsamples = learning['n_ways'] * learning['n_shot']
    n_usamples = learning['n_ways'] * learning['n_queries']
    n_samples = n_lsamples + n_usamples
    
    print('Start classification for %d tasks...'%(n_runs))
    ndatas = GenerateRunSet(end=n_runs, cfg=cfg, data = data, mode = 'Val') #(n_runs, n_samples, 2048)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
    5).clone().view(n_runs, n_samples)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
                    5).clone().view(n_runs, n_samples)

    acc_list = []
    f1_list = []
    recall_list = []
    test_acc_list = []
    A = True
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(n_gaus/n_shot)
        model_copy = model
        train_test(model_copy, support_data, support_label, learning, my_loss, optimizer)
        for j in range(n_lsamples):
            #模型先训练好再放进来采样
            x_plus, graphs, _ = model_copy(support_data[j], mode = "Val", n_gaus = num_sampled)

            sampled_data.append(x_plus.cpu().detach().numpy())
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = sampled_data
        Y_aug = sampled_label
        # ---- train classifier
        classifier = LogisticRegression(max_iter=2000).fit(X=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        RIM_pre = []
        out_x, kl_g, kl_b, graph, _ = model_copy(query_data[0], mode = "Te")
        out_x = out_x.unsqueeze(0)
        for i in range(1, query_data.shape[0]):
            out_x1, kl_g, kl_b, graph, _ = model_copy(query_data[i], mode = "Te")
            
            out_x1 = out_x1.unsqueeze(0)
            out_x = torch.cat((out_x, out_x1), dim = 0)
            
            RIM_pre.append(out_x)
        pred = torch.argmax(out_x.data, 1)
        predicts2 = pred.cpu().detach().numpy()


        acc = np.mean(predicts == query_label)
        f1 = f1_score(query_label, predicts, average = 'macro')
        recall = recall_score(query_label, predicts, average = None)
        acc_list.append(acc)
        f1_list.append(f1)
        recall_list.append(recall)
    print(acc_list)
    acc_ci = 1.96 * float(np.std(acc_list) / np.sqrt(len(acc_list)))
    print('derm: %d way %d shot  ACC : %f +/- %f;  F1 : %f ;  Recall : %f'%(n_ways,n_shot,float(np.mean(acc_list)), acc_ci, float(np.mean(f1_list)), float(np.mean(recall_list))))
    return np.mean(acc_list)

def test_FSL(model, data, learning, epoch, my_loss, optimizer, n_runs = 20):
    cfg = {'shot': learning['n_shot'], 'ways': learning['n_ways'], 'queries': learning['n_queries'], 'feature_dim': learning['feature_dim']}
    n_gaus = learning['n_gaus']
    n_shot = learning['n_shot']
    n_ways = learning['n_ways']
    n_queries = learning['n_queries']
    n_lsamples = learning['n_ways'] * learning['n_shot']
    n_usamples = learning['n_ways'] * learning['n_queries']
    n_samples = n_lsamples + n_usamples
    
    print('Start classification for %d tasks...'%(n_runs))
    ndatas = GenerateRunSet(end=n_runs, cfg=cfg, data = data, mode = 'Val') #(n_runs, n_samples, 2048)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
    5).clone().view(n_runs, n_samples)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 
                    5).clone().view(n_runs, n_samples)

    acc_list = []
    f1_list = []
    recall_list = []
    test_acc = []
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(n_gaus/n_shot)
        model_copy = model
        train_test(model_copy, support_data, support_label, learning, my_loss, optimizer)
        for j in range(n_lsamples):
            #模型先训练好再放进来采样
            x_plus, graphs,_ = model_copy(support_data[j], mode = "Val", n_gaus = num_sampled)

            sampled_data.append(x_plus.cpu().detach().numpy())
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = sampled_data
        Y_aug = sampled_label
        # ---- train classifier
        classifier = LogisticRegression(max_iter=2000).fit(X=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        RIM_pre = []
        out_x, kl_g, kl_b, graph,_ = model_copy(query_data[0], mode = "Te")
        out_x = out_x.unsqueeze(0)
        for i in range(1, query_data.shape[0]):
            out_x1, kl_g, kl_b, graph,_ = model_copy(query_data[i], mode = "Te")
            
            out_x1 = out_x1.unsqueeze(0)
            out_x = torch.cat((out_x, out_x1), dim = 0)
            
            RIM_pre.append(out_x)
        pred = torch.argmax(out_x.data, 1)
        predicts2 = pred.cpu().detach().numpy()


        acc = np.mean(predicts == query_label)

        f1 = f1_score(query_label, predicts, average = 'macro')
        recall = recall_score(query_label, predicts, average = None)
        acc_list.append(acc)
        f1_list.append(f1)
        recall_list.append(recall)
    print(graphs[-1])
    print(acc_list)
    acc_ci = 1.96 * float(np.std(acc_list) / np.sqrt(len(acc_list)))
    print('derm: %d way %d shot  ACC : %f +/- %f;  F1 : %f ;  Recall : %f'%(n_ways,n_shot,float(np.mean(acc_list)), acc_ci,float(np.mean(f1_list)), float(np.mean(recall_list))))
    return np.mean(acc_list)

def GenerateRunSet(cfg, data, start = 0, end = 100, mode = 'Test'):
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 1}

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], cfg['feature_dim']))
    for iRun in range(end-start):
        dataset[iRun] = data.GenerateRun(start+iRun, cfg, mode) 
    return dataset
