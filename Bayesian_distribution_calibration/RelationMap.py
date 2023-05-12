import torch.nn as nn
import numpy as np
import torch
import copy
from base_func import *
from GCN import GCN
import torch.nn.functional as F




class encode_mean_std_pair(nn.Module):
    def __init__(self, graph_node_dim, h_dim, dropout=0.1):
        super(encode_mean_std_pair, self).__init__()

        self.graph_node_dim = graph_node_dim
        self.h_dim = h_dim
        self.dropout = dropout

        self.enc = nn.Sequential(nn.Linear(graph_node_dim, h_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.enc_g = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))
        self.enc_b = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout))

        self.enc_mean = nn.Linear(h_dim, 1)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, 1), nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        mean = self.enc_mean(enc_g)
        std = self.enc_std(enc_g)
        return mean, std, enc_b



"""
    输入base嵌入和需要推断关联的嵌入，推断新嵌入和base嵌入的关系，
    训练一个采样生成器，可以加一些对比学习的方法
"""

class RIM(nn.Module):
    def __init__(self, feature_dim, base_num, edge_dim, base_nodes, n_gaus, last_dense = 128, out_dim = 64, g_dim = 512, dropout = 0.1):
        """
        feature_dim:输入特征长度
        base_num:base node类别数
        base_nodes:base_num * feature_dim, 基类节点嵌入
        n_gaus:高斯采样数，即生成的采样特征数
        """
        super(RIM, self).__init__()
        

        print("initializing RIM model for %d base, %d feature"%(base_num, feature_dim))
        self.feature_dim = feature_dim
        self.base_num = base_num
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.base_nodes = torch.from_numpy(base_nodes).cuda()
        #self.base_means = []
        #self.base_cov = []
        self.truee = True
        self.true2 = True
        self.n_gaus = n_gaus
        self.g_dim = g_dim
        self.out_dim = out_dim
        self.last_dense = last_dense
        self.ture3 = True
        """
        for base_class in self.base_nodes:
            base_feature = np.array(base_class)
            mean = np.mean(base_feature, axis = 0)
            cov = np.cov(base_feature.T)
            self.base_means.append(mean)
            self.base_cov.append(cov)
        """

        # recurrence
        self.gcn1 = GCN(self.feature_dim, self.g_dim, num_node=base_num, input_vector=True)
        #self.gcn2 = GCN(self.g_dim, self.g_dim, num_node=base_num, input_vector=False)

        # prior
        self.prior_enc = encode_mean_std_pair(self.edge_dim,self.edge_dim, self.dropout_rate)
        self.prior_mij = nn.Linear(self.edge_dim, 1)

        # post
        self.post_enc = encode_mean_std_pair(self.edge_dim, self.edge_dim,self.dropout_rate)
        self.post_mean_approx_g = nn.Linear(self.edge_dim, 1)
        self.post_std_approx_g = nn.Sequential(nn.Linear(self.edge_dim, 1), nn.Softplus())


        #边嵌入生成图嵌入
        self.post_emb_to_graph = nn.Sequential(
            nn.Linear(self.edge_dim, self.edge_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.edge_dim, 1), nn.ReLU()
        )


        #边嵌入生成
        self.gen_edge_emb = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.edge_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.edge_dim, self.edge_dim)
        )

        #edge2graph
        self.graph_trans = nn.Sequential(
            nn.Linear(self.edge_dim, 1),
            nn.ReLU()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(self.g_dim, self.last_dense),
            #nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.last_dense, self.out_dim)
        )
        print(self.out_dim)





    def forward(self, x, mode = "Train", n_gaus = 1000):
        #b, _ = x.shape #batchsize*feature_dim
        self.n_gaus = n_gaus
        if mode == "Train":
            x_node_emb = x.clone()
        else:
            x_node_emb = torch.from_numpy(x).cuda()
        #print(x_node_emb.shape)
        #x_node_emb = x_node_emb.reshape(b, -1)
        #x_node_emb = x.clone() #target_node
        node_pairs = torch.zeros(self.base_num, self.feature_dim * 2).cuda()
        #for j in range(b):
        for i in range(self.base_num):
            node_pairs[i] = torch.cat([self.base_nodes[i], x_node_emb], dim = 0) #用于计算边嵌入

        

        #node2edge
        edge_emb = self.gen_edge_emb(node_pairs) #边嵌入生成(b*base_num*edge_dim)

        input4prior = edge_emb.clone()
        input4post = edge_emb.clone()

        #先验图
        prior_mean, prior_std, prior_b = self.prior_enc(input4prior)
        prior_mij = self.prior_mij(prior_b)
        #print(prior_mij.shape)
        prior_mij = 0.4 * sigmoid(prior_mij)
        prior_mij = prior_mij.squeeze() #batch*base_num
        #print(prior_mij.shape)

        #后验图
        post_mean_g,post_std_g,post_b = self.post_enc(input4post)
        post_mean_approx_g = self.post_mean_approx_g(post_b) # (W,F_g)
        post_std_approx_g = self.post_std_approx_g(post_b)   # (W,F_g)
        post_mean_approx_g = post_mean_approx_g.squeeze()
        post_std_approx_g = post_std_approx_g.squeeze()

        # estimate post mij for Binomial Dis
        nij = softplus(post_mean_approx_g) + 0.01
        nij_ = 2.0 * nij * post_std_approx_g.pow(2)
        post_mij = 0.5 * (1.0 + nij_ - torch.sqrt(nij_.pow(2) + 1))
        post_mean_g = post_mean_g.squeeze()
        post_std_g = post_std_g.squeeze()

        #print(post_mij.shape)
        #print(post_mean_g.shape)
        #print(post_std_g.shape) 

        alpha_bars, alpha_tilde = self.sample_repara(post_mean_g, post_std_g, post_mij, self.n_gaus, mode)
        if self.truee:
            print(alpha_bars)
            print(alpha_tilde)
            
            self.truee = False

        # regularization
        a1 = alpha_tilde * post_mean_g
        a2 = torch.sqrt(alpha_tilde) * post_std_g
        a3 = alpha_tilde * prior_mean
        a4 = torch.sqrt(alpha_tilde) * prior_std

        kl_g = self.kld_loss_gauss(a1, a2, a3, a4)
        kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)

        #kl_g, S_kl_g = self.kld_loss_gauss(a1, a2, a3, a4)
        #print(post_mij.shape)
        #print(prior_mij.shape)
        #kl_b, S_kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)
        #print(alpha_bars.shape)
        #print(x.shape)
        #x_torch = torch.from_numpy(x)
        node_features = torch.zeros(self.base_num, self.feature_dim).cuda()
        for i in range(self.base_num):
            node_features[i] = self.base_nodes[i]
        #alpha_bars = F.softmax(alpha_bars,dim = -1)
        #alpha_bars = F.softmax(alpha_bars,dim = -1)#alpha_bars + 1#
        H_g= torch.matmul(alpha_bars, node_features)#self.gcn1(node_features, alpha_bars) 
        H_g = H_g + x_node_emb
        #H_g, _ = self.gcn2(H_g, A)
        #print(H_g.shape)
        if mode == "Train":
            x_out = self.out_layer(H_g)
            return x_out, kl_g, kl_b, alpha_bars
        else:
            return H_g, alpha_bars #n_gaus * g_dim
    

    #距离图生成，边强度为中心距离的倒数（暂定）
    def simDis(self, mean, std, node, k = 2, alpha = 0.21):
        dist = []
        dist_graph = torch.zeros(self.base_num) 

        for i in range(len(self.base_means)):
            dist.append(np.linalg.norm(mean-self.base_means[i]))
            dist_graph[i] = 1/dist[i]

        index = np.argpartition(dist, k)[:k]
        means = np.concatenate([np.array(self.base_means)[index], mean])
        covs = np.concatenate([np.array(self.base_cov)[index], std])
        calibrated_mean = np.mean(means, axis=0)
        calibrated_cov = np.mean(covs, axis=0)+alpha

        return calibrated_mean, calibrated_cov, dist_graph

    def sample_repara(self, mean, std, mij, n_gaus, mode):
        mean_alpha = mij
        '''Poisson'''
        #std_alpha = torch.sqrt(mij)ex
        #print("input shapes:")
        #print(mean.shape)
        #print(std.shape)
        #print(mij.shape)

        #print("others")
        std_alpha = torch.sqrt(mij*(1.0 - mij))
        #print(std_alpha.shape)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps * std_alpha + mean_alpha
        #print(alpha_tilde.shape)
        alpha_tilde = softplus(alpha_tilde)#*self.prior_graph

        mean_sij = alpha_tilde * mean
        std_sij = torch.sqrt(alpha_tilde) * std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = eps_2 * std_sij + mean_sij
        alpha_bar = softplus(s_ij * alpha_tilde)
        #alpha_bar = self.norm(alpha_bar)
        if mode == "Train":
            return alpha_bar, alpha_tilde
        #print(alpha_bar.shape)
        alpha_bars = torch.zeros(n_gaus, alpha_bar.shape[-1]).cuda()
        #alpha_bar = self.my_norm(alpha_bar)
        alpha_bars[0] = alpha_bar
        if self.ture3:
            print(mean)
            print(std)
            print(mij)
            print("end")
            self.ture3 = False
        for i in range(1, n_gaus):
            eps_i = torch.FloatTensor(std.size()).normal_().cuda()
            s_ij = eps_i * std_sij + mean_sij
            alpha_bars[i] = softplus(s_ij * alpha_tilde)#self.my_norm(softplus(s_ij * alpha_tilde))
        #print(alpha_bars.shape)

        return alpha_bars, alpha_tilde
    
    def my_norm(self, x):
        if self.true2:
            print("ggggg")
            print(torch.min(x))
            print(torch.max(x))
            print("sdsdsdsd")
            self.true2 = False
        return x - torch.min(x) / torch.max(x) - torch.min(x)
    
    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior, eps=1e-6): #高斯部分的散度
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        return 0.5 * torch.sum(torch.abs(kld_element))

    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior, eps=1e-6): #伯努利部分的ELBO
        kld_element = mij_prior - mij_post + \
                       mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        return torch.sum(torch.abs(kld_element))

    """
    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior):
        eps = 1e-6
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        #print(kld_element.shape)
        kld_element = kld_element[:, -25:, :]
        #print("klg:", kld_element.shape)
        return 0.05 * torch.sum(torch.abs(kld_element)), 0.5 * kld_element
    
    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior):
        eps = 1e-6
        '''Poisson'''
        '''
        kld_element_term1 = mij_prior - mij_post + \
                            mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        #'''
        first_item = mij_post*(torch.log(mij_post+eps)-torch.log(mij_prior+eps))
        second_item = (1-mij_post)*(torch.log(1-mij_post+0.5*mij_post.pow(2)+eps)-
                                    torch.log(1-mij_prior+0.5*mij_prior.pow(2)+eps))
        kld_element_term1 = first_item + second_item
        #'''
        #print("*"*128)
        #print(kld_element_term1.shape)
        kld_element_term1 = kld_element_term1[:, -25:, ]
        #print(kld_element_term1.shape)
        return 0.1 * torch.sum(torch.abs(kld_element_term1)), kld_element_term1
    """



class TestModel(nn.Module):
    def __init__(self, feature_dim, base_num, edge_dim, base_nodes, n_gaus, last_dense = 128, out_dim = 64, g_dim = 2048, dropout = 0.1):
        """
        feature_dim:输入特征长度
        base_num:base node类别数
        base_nodes:base_num * feature_dim, 基类节点嵌入
        n_gaus:高斯采样数，即生成的采样特征数
        """
        super(TestModel, self).__init__()
        

        print("initializing RIM model for %d base, %d feature"%(base_num, feature_dim))
        self.feature_dim = feature_dim
        self.base_num = base_num
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.base_nodes = torch.from_numpy(base_nodes).cuda()
        #self.base_means = []
        #self.base_cov = []
        self.n_gaus = n_gaus
        self.g_dim = g_dim
        self.out_dim = out_dim
        self.last_dense = last_dense

        self.out_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.last_dense * 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.last_dense * 2, self.last_dense),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.last_dense, self.out_dim)
        )
        print(self.out_dim)





    def forward(self, x, mode = "Train"):
        #b, _ = x.shape #batchsize*feature_dim
        if mode == "Train":
            x_node_emb = x.clone()
        else:
            x_node_emb = torch.from_numpy(x).cuda()
        return self.out_layer(x_node_emb), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        