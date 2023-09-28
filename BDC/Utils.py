import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

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

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, _= x.size()

        optimizer.zero_grad()
        out_x, kl_g, kl_b, graph = model(x[0])
        out_x = out_x.unsqueeze(0)

        for i in range(1,b):
            out_x1, kl_g1, kl_b1, _ = model(x[i])
            kl_g = kl_g + kl_g1
            kl_b = kl_b + kl_b1
            out_x1 = out_x1.unsqueeze(0)
            out_x = torch.cat((out_x, out_x1), dim = 0)
        #print(x.shape)
        #print(out_x.shape)
        #torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

        loss_kl1 = learning['lamada1'] * kl_g
        loss_kl2 = learning['lamada2'] * kl_b
        loss = my_loss(out_x, y) + loss_kl1 + loss_kl2
        #print(loss)
        loss.backward()
        #with torch.autograd.detect_anomaly():
        #    loss.backward()
        #print(graph)

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        pred = torch.argmax(out_x.data, 1)
        #print(pred.shape)
        #if learning['smooth']:
        #    y = torch.argmax(y, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        total += len(y)
        loss_list.append(loss.item())
        loss_kl1_list.append(loss_kl1.item())
        loss_kl2_list.append(loss_kl2.item())
        #print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #      (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    print(graph)
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    print("[TR]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total)))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean


def val(model, val_loader, learning, my_loss, optimizer, epoch, save = False):
    model.eval()
    acc = 0
    total = 0
    loss_list = []
    pred_list = []
    true_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
        # If you run this code on CPU, please remove the '.cuda()'
            x = x.cuda()
            y = y.cuda()
            b, _= x.size()

            optimizer.zero_grad()
            out_x, kl_g, kl_b = model(x[0])
            out_x = out_x.unsqueeze(0)

            for i in range(1,b):
                out_x1, kl_g1, kl_b1 = model(x[i])
                kl_g = kl_g + kl_g1
                kl_b = kl_b + kl_b1
                out_x1 = out_x1.unsqueeze(0)
                out_x = torch.cat((out_x, out_x1), dim = 0)

            loss_kl1 = learning['lamada1'] * kl_g
            loss_kl2 = learning['lamada2'] * kl_b
            loss = my_loss(out_x, y) + loss_kl1 + loss_kl2

            pred = torch.argmax(out_x.data, 1)
            #print(pred.shape)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.cpu().numpy())
            acc += ((pred == y).sum()).cpu().numpy()
            total += len(y)
            loss_list.append(loss.item())
            loss_kl1_list.append(loss_kl1.item())
            loss_kl2_list.append(loss_kl2.item())
            #print("[VAL]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
            #      (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    print("[VA]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total)))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean


def val_FSL(model, data, learning, epoch, n_runs = 20):
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
    #print(labels.shape)
    #print(ndatas.shape)

    acc_list = []
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(n_gaus/n_shot)
        for j in range(n_lsamples):
            #模型先训练好再放进来采样
            x_plus, graphs = model(support_data[j], mode = "Val", n_gaus = num_sampled)
            sampled_data.append(x_plus.cpu().detach().numpy())
            #print(x_plus.cpu().detach().numpy().shape)
            #mean, cov = distribution_calibration(support_data[j], base_means, base_cov, k=2)
            #sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        #print("sample:", sampled_data.shape)
        X_aug = sampled_data
        Y_aug = sampled_label
        # ---- train classifier
        classifier = LogisticRegression(max_iter=2000).fit(X=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
    print(graphs[-1])
    print('derm: %d way %d shot  ACC : %f'%(n_ways,n_shot,float(np.mean(acc_list))))
    return np.mean(acc_list)

def test_FSL(model, data, learning, n_runs = 20):
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
    #print(labels.shape)
    #print(ndatas.shape)

    acc_list = []
    for i in tqdm(range(n_runs)):
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(n_gaus/n_shot)
        for j in range(n_lsamples):
            #模型先训练好再放进来采样
            x_plus, graphs = model(support_data[j], mode = "Val", n_gaus = num_sampled)
            #print(graphs[0])
            sampled_data.append(x_plus.cpu().detach().numpy())
            #sampled_data.append(model(support_data[j], mode = "Val").cpu().detach().numpy())
            #mean, cov = distribution_calibration(support_data[j], base_means, base_cov, k=2)
            #sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_label[j]]*num_sampled)
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
        X_aug = sampled_data
        Y_aug = sampled_label
        # ---- train classifier
        classifier = LogisticRegression(max_iter=2000).fit(X=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
    print('derm: %d way %d shot  ACC : %f'%(n_ways,n_shot,float(np.mean(acc_list))))
    return np.mean(acc_list)

def GenerateRunSet(cfg, data, start = 0, end = 100, mode = 'Test'):
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 1}

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], cfg['feature_dim']))
    for iRun in range(end-start):
        dataset[iRun] = data.GenerateRun(start+iRun, cfg, mode) #(ways, shot+queries, data.shape)
    #print(dataset.shape)
    return dataset




def train_epoch2(model, train_loader, learning, my_loss, optimizer, epoch, threshold=10):
    model.train()
    acc = 0
    total = 0
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        #print(y)
        b, _= x.size()

        optimizer.zero_grad()
        out_x, kl_g, kl_b = model(x)
        loss_kl1 = learning['lamada1'] * kl_g
        loss_kl2 = learning['lamada2'] * kl_b
        loss = my_loss(out_x, y) + loss_kl1 + loss_kl2
        #print(loss)
        loss.backward()
        #with torch.autograd.detect_anomaly():
        #    loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        #print("#######################")
        #print(out_x.data.shape)
        pred = torch.argmax(out_x.data, 1)
        #print(pred.shape)
        #print(y.shape)
        #if learning['smooth']:
        #    y = torch.argmax(y, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        total += len(y)
        loss_list.append(loss.item())
        loss_kl1_list.append(loss_kl1.item())
        loss_kl2_list.append(loss_kl2.item())
        #print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #      (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    print("[TR]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total)))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean




def draw_graph(model, data, learning, n_runs = 20):
    graph_data = np.array(data.GenAll())
    #print(graph_data.shape)
    
    x_plus, graphs = model(graph_data[0][0], mode = "Val", n_gaus = 10)
    g1 = graphs
    print(graphs.shape)
    #print(graph_data[0][0])
    #for data in graph_data[0]:
    ##    x_plus, graph = model(data, mode = "Val", n_gaus = 10)
    #   graphs = graphs + graph
    #graphs = graphs - g1
    #graphs = graphs / graph_data.shape[-2]
    return graphs