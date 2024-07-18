import torch.nn as nn
import numpy as np
import torch
import copy
from base_func import *
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




class RIM(nn.Module):
    def __init__(self, feature_dim, base_num, edge_dim, base_nodes, n_gaus, last_dense = 128, out_dim = 64, g_dim = 512, dropout = 0.1):
        """
        feature_dim:input feature dimension
        base_num:number of base classes
        base_nodes:base_num * feature_dim, base node features
        n_gaus: number of Gaussian graph samples
        """
        super(RIM, self).__init__()
  
        
        self.feature_dim = feature_dim
        self.base_num = base_num
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.base_nodes = torch.from_numpy(base_nodes).cuda()
        self.n_gaus = n_gaus
        self.g_dim = g_dim
        self.out_dim = out_dim
        self.last_dense = last_dense

        self.x_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(p=0.1)
        )
        
        # prior
        self.prior_enc = encode_mean_std_pair(self.edge_dim,self.edge_dim, self.dropout_rate)
        self.prior_mij = nn.Linear(self.edge_dim, 1)

        # post
        self.post_enc = encode_mean_std_pair(self.edge_dim, self.edge_dim,self.dropout_rate)
        self.post_mean_approx_g = nn.Linear(self.edge_dim, 1)
        self.post_std_approx_g = nn.Sequential(nn.Linear(self.edge_dim, 1), nn.Softplus())
        self.truee = True
        self.ture3 = True


        #from edge embeddings to graph
        self.post_emb_to_graph = nn.Sequential(
            nn.Linear(self.edge_dim, self.edge_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.edge_dim, 1), nn.ReLU()
        )


        #edge embedding generation
        self.gen_edge_emb = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.edge_dim), nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.edge_dim, self.edge_dim)
        )

        #classification layer of training phase
        self.out_layer = nn.Sequential(
            nn.Linear(self.g_dim, self.last_dense),
            #nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.last_dense, self.out_dim)
        )

        #classification layer for few-shot tasks
        self.out_layer_fs = nn.Sequential(
            nn.Linear(self.g_dim, self.last_dense),
            #nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.last_dense, 5)
        )


    def forward(self, x, mode = "Train", n_gaus = 1000):
        self.n_gaus = n_gaus

        #Convert input features to torch array format
        if mode == "Train":
            x_node_emb = x.clone()
        elif mode == "Te":
            x_node_emb = torch.from_numpy(x).cuda()
        elif mode == "TV":
            x_node_emb = x.clone()
        else:
            x_node_emb = torch.from_numpy(x).cuda()
        
        x_node0 = x_node_emb
        base_nodes = self.base_nodes
        
        node_pairs = torch.zeros(self.base_num, self.feature_dim * 2).cuda()
        
        for i in range(self.base_num):
            node_pairs[i] = torch.cat([base_nodes[i], x_node_emb], dim = 0) #for edge embedding calculation
        x_node_emb = self.x_linear(x_node_emb)

        #node2edge
        edge_emb = self.gen_edge_emb(node_pairs) #edge embedding generation(b*base_num*edge_dim)


        input4prior = edge_emb.clone()
        input4post = edge_emb.clone()

        #prior graph
        prior_mean, prior_std, prior_b = self.prior_enc(input4prior)
        prior_mij = self.prior_mij(prior_b)
        prior_mij = 0.4 * sigmoid(prior_mij)
        prior_mij = prior_mij.squeeze() #batch*base_num

        #post graph
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

        alpha_bars, alpha_tilde = self.sample_repara(post_mean_g, post_std_g, post_mij, self.n_gaus, mode)

        # regularization
        a1 = alpha_tilde * post_mean_g
        a2 = torch.sqrt(alpha_tilde) * post_std_g
        a3 = alpha_tilde * prior_mean
        a4 = torch.sqrt(alpha_tilde) * prior_std

        kl_g = self.kld_loss_gauss(a1, a2, a3, a4)
        kl_b = self.kld_loss_binomial_upper_bound(post_mij, prior_mij)

        node_features = torch.zeros(self.base_num, self.feature_dim).cuda()
        for i in range(self.base_num):
            node_features[i] = base_nodes[i]
        H_g= torch.matmul(alpha_bars, node_features)
        H_g_abs = torch.abs(H_g)
        if mode == "Train":
            H_g = 0.5 * torch.div(H_g, H_g_abs.sum(dim = -1)) + 0.5 * x_node0 
            x_out = self.out_layer(H_g)
            return x_out, kl_g, kl_b, alpha_bars, H_g
        elif mode == "TV" or mode == "Te":
            H_g = 0.5 * torch.div(H_g, H_g_abs.sum(dim = -1)) + 0.5 * x_node0
            x_out = self.out_layer_fs(H_g)
            return x_out, kl_g, kl_b, alpha_bars, H_g
        else:
            H_g_sum = H_g_abs.sum(dim = -1)
            H_g = torch.transpose(H_g, 0, 1)
            H_g = torch.div(H_g, H_g_sum)
            H_g = torch.transpose(H_g, 0, 1)
            H_g = 0.5 * H_g + 0.5 * x_node0 
            return H_g, alpha_bars[0], H_g 


    def sample_repara(self, mean, std, mij, n_gaus, mode):
        mean_alpha = mij
        std_alpha = torch.sqrt(mij*(1.0 - mij))
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps * std_alpha + mean_alpha
        alpha_tilde = softplus(alpha_tilde)

        mean_sij = alpha_tilde * mean
        std_sij = torch.sqrt(alpha_tilde) * std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = 0.01 * eps_2 * std_sij + mean_sij
        alpha_bar = s_ij * alpha_tilde
        if mode == "Train" or mode == "TV" or mode == "Te":
            return self.norm(alpha_bar), alpha_tilde
        alpha_bars = torch.zeros(n_gaus, alpha_bar.shape[-1]).cuda()
        alpha_bars[0] = self.norm(alpha_bar)
        
        for i in range(1, n_gaus):
            eps_i = torch.FloatTensor(std.size()).normal_().cuda()
            alpha_bars[i] = self.norm(s_ij * alpha_tilde)

        return alpha_bars, alpha_tilde
    
    def norm(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    
    #kl loss of Gaussian distribution estimate
    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior, eps=1e-6): 
        kld_element = (2 * torch.log(std_prior + eps) - 2 * torch.log(std_post + eps) +
                       ((std_post).pow(2) + (mean_post - mean_prior).pow(2)) /
                       (std_prior + eps).pow(2) - 1)
        return 0.5 * torch.sum(torch.abs(kld_element))

    #ELBO
    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior, eps=1e-6): 
        kld_element = mij_prior - mij_post + \
                       mij_post * (torch.log(mij_post+eps) - torch.log(mij_prior+eps))
        return torch.sum(torch.abs(kld_element))