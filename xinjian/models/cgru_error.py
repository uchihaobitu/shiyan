# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:01:33 2022

@author: 61995
"""


import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from metrics.visualization_metrics import visualization
from tqdm import tqdm
from .linear import notears_linear
from torch.autograd import Function


import torch.optim as optim


class GRU(nn.Module):
    def __init__(self, num_series, hidden):

        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        #Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               

    def forward(self, X, z, connection, mode = 'train'):
        
        # X=X[:,:,np.where(connection!=0)[0]]
        indices = torch.where(connection != 0)[0]
        X = X[:, :, indices]
        device = self.gru.weight_ih_l0.device
        tau = 0
        if mode == 'train':
          X_right, hidden_out = self.gru(torch.cat((X[:,0:1,:],X[:,11:-1,:]),1), z)
          X_right = self.linear(X_right)

          return X_right, hidden_out
          
class VRAE4E(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        Error VAE
        '''
        super(VRAE4E, self).__init__()
        self.device = torch.device('cuda:1')
        # self.device = device
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)#nn.Linear(hidden, 1)
        self.fc_std = nn.Linear(hidden, hidden)
        
        self.linear_hidden = nn.Linear(hidden, hidden)
        self.tanh = nn.Tanh()
        
        
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, num_series)
        

        


    def init_hidden(self, batch):
        '''Initialize hidden states for GRU cell.'''
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               

    def forward(self, X, mode = 'train'):
        
        X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
        if mode == 'train':
            

            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:,1:,:], hidden_0.detach())
            
            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)
            
            sigma = torch.exp(0.5*log_var)
            z = torch.randn(size = mu.size())
            z = z.type_as(mu) 
            z = mu + sigma*z
            z = self.tanh(self.linear_hidden(z))

            
            X_right, hidden_out = self.gru(X[:,:-1,:], z)

            pred = self.linear(X_right)
            
            

            return pred, log_var, mu
        if mode == 'test':

            X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
            h_t = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
            for i in range(int(20/1)+1):
                out, h_t = self.gru(X_seq[:,-1:,:], h_t)
                out = self.linear(out)
                #out = self.sigmoid(out)
                X_seq = torch.cat([X_seq,out],dim = 1)
            return X_seq
        

          


            
                
          


class CRVAE(nn.Module):
    def __init__(self, num_series, connection, hidden):
        '''
        connection: pruned networks
        '''
        super(CRVAE, self).__init__()
        
        self.device = torch.device('cuda:1')
        # self.device = device
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        self.connection = connection

        # Set up networks.
        self.networks = nn.ModuleList([
            GRU(int(connection[:,i].sum()), hidden) for i in range(num_series)])

    def forward(self, X, noise = None, mode = 'train', phase = 0):

        if phase == 0:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                
    
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu)
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
    
                return pred, log_var, mu
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t
                    else:
                        X_seq = torch.cat([X_seq,X_t],dim = 1)
                        
                    #out = self.sigmoid(out)
                    
                return X_seq
            
        
        if phase == 1:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                
    
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
                
                
    
                return pred, log_var, mu
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t + 0.1*noise[:,i:i+1,:] 
                    else:
                        X_seq = torch.cat([X_seq,X_t+0.1*noise[:,i:i+1,:]],dim = 1)
                        
                    #out = self.sigmoid(out)
                    
                return X_seq
        

    def GC(self, threshold=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        #print(GC)
        if threshold:
            return (torch.abs(GC) > 0).int()
        else:
            return GC





def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.gru.flatten_parameters()





def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    return lam * (
        torch.sum(network.linear.weight ** 2) +
        torch.sum(network.gru.weight_hh_l0 ** 2))# + 
        #torch.sum(network.fc_std.weight ** 2) + 
        #torch.sum(network.fc_mu.weight ** 2) + 
        #torch.sum(network.fc_std.weight ** 2))



def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(len(data) - context, context, data.shape[1],
                        dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input.detach(), target.detach()


def MinMaxScaler(data):
  """Min-Max Normalizer.
  
  Args:
    - data: raw data
    
  Returns:
    - norm_data: normalized data
    - min_val: minimum values (for renormalization)
    - max_val: maximum values (for renormalization)
  """    
  min_val = np.min(np.min(data, axis = 0), axis = 0)
  data = data - min_val
    
  max_val = np.max(np.max(data, axis = 0), axis = 0)
  norm_data = data / (max_val + 1e-7)
    
  return norm_data


def train_phase2(crvae, vrae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 1024):
    '''Train model with Adam.'''
    optimizer = optim.Adam(vrae.parameters(), lr=1e-3)
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    
    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    
    Y = Y_all[idx]
    X_v = X_all[batch_size:]
    start_point = 0#context-10-1
    beta = 1#0.001
    beta_e = 1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate smooth error.
    pred,mu,log_var = crvae(X)#


    
    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    
    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd
    
    
    error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
    pred_e,mu_e,log_var_e = vrae(error)
    loss_e = loss_fn(pred_e, error)
    mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    smooth_e = loss_e + beta_e*mmd_e

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth_e.backward()
        if lam == 0:
            optimizer.step()
            optimizer.zero_grad()  
        
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        

        crvae.zero_grad()

        # Calculate loss for next iteration.
        idx = np.random.randint(len(X_all), size=(batch_size,))
    
        #X = X_all[idx]
       
        #Y = Y_all[idx]
        
        pred,mu,log_var = crvae(X)#
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd
        
        
        error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
        pred_e,mu_e,log_var_e = vrae(error)
        loss_e = loss_fn(pred_e, error)
        mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        smooth_e = loss_e + beta_e*mmd_e
        


        # Check progress.
        if (it) % check_every == 0:

            
            
            X_t = X
            pred_t,mu_t ,log_var_t= crvae(X_t)
            
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
        

            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)
                

                print('Loss_e = %f' % smooth_e)
                print('KL_e = %f' % mmd_e)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))


            
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)


                
            start_point = 0
            predicted_error = vrae(error, mode = 'test').detach()
            
            predicted_data = crvae(X_t, predicted_error, mode = 'test', phase = 1)
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            

            if it % 1000 ==0:
                plt.plot(ori[0,:,1])
                plt.plot(syn[0,:,1])
                plt.show()

                visualization(ori, syn, 'pca')
                visualization(ori, syn, 'tsne')
                np.save('ori_henon.npy',ori)
                np.save('syn_henon.npy',syn)


    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list


def train_phase1(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 2048):
    '''Train model with Adam.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    

    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    
    Y = Y_all[idx]
    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate crvae error.
    pred,mu,log_var = crvae(X)

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd
    

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)
        
        # 计算迹指数损失并加入到总损失中
        adjacency_matrix = crvae.GC(threshold=False)
      
        total_loss = loss + ridge + beta * mmd  # 将cycle_loss加入到总损失中
        

        crvae.zero_grad()


        pred,mu,log_var = crvae(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])

        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd
        

        # Check progress.
        if (it) % check_every == 0:     
            X_t = X
            Y_t = Y
            
            pred_t,mu_t ,log_var_t= crvae(X_t)
            
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
        

            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))



            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)

                
            start_point = 0
            predicted_data = crvae(X_t,mode = 'test')
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            
            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list




def test_models(crvae, vrae, X_test, device):
    """
    对给定的测试数据进行预测，并计算预测序列与真实序列之间的MSE。

    Args:
    - crvae: 训练好的CRVAE模型。
    - vrae: 训练好的VRAE4E模型。
    - X_test: 测试数据，形状为(batch_size, sequence_length, num_features)。
    - device: 使用的设备（'cuda'或'cpu'）。

    Returns:
    - pred_sequences: 预测的序列，用于后续的性能评估。
    """
    crvae.eval()
    vrae.eval()

    # 确保测试数据在正确的设备上
    X_test = X_test.to(device)

    # 使用CRVAE模型生成测试序列的预测
    pred_sequences_crvae, _, _ = crvae(X_test, mode='test')

    # 使用VRAE4E模型对CRVAE的输出进行进一步的处理（如果需要）
    # 注意：这里假设两个模型的输出和输入是兼容的，您可能需要根据实际情况调整
    pred_sequences = vrae(pred_sequences_crvae, mode='test')

    return pred_sequences


def calculate_cycle_loss(adjacency_matrix, lam_cycle):
    exp_A = torch.matrix_exp(adjacency_matrix - torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device))
    trace_exp_A = torch.diagonal(exp_A).sum()
    cycle_loss = lam_cycle * (trace_exp_A - adjacency_matrix.size(0))
    return cycle_loss


def train_phase3(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                 check_every=50, verbose=1, sparsity=100, batch_size=2048, lam_cycle=0.01,beta=1):
    '''Train model with Adam and ensure causality matrix is acyclic.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Initialize optimizer
    optimizer = optim.Adam(crvae.parameters(), lr=lr)
    
    for it in range(max_iter):
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]
        Y_batch = Y_all[idx]

        # Forward pass
        pred, mu, log_var = crvae(X_batch)
        loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) for i in range(p)])
        
        # Calculate regularization terms
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim=-1).sum(dim=0)).mean(dim=0)
        ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        smooth_loss = loss + ridge + beta*mmd
        
        # Calculate cycle loss
        adjacency_matrix = crvae.GC(threshold=False)
        cycle_loss = calculate_cycle_loss(adjacency_matrix, lam_cycle)

        # Total loss
        total_loss = smooth_loss + cycle_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        total_loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        # Log the training process
        if (it) % check_every == 0:
            with torch.no_grad():
                current_loss = total_loss.item()
                print(f'Iteration {it}, Total Loss: {current_loss:.4f}')
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_it = it
                    best_model = deepcopy(crvae.state_dict())
    
    # Restore best model at the end
    crvae.load_state_dict(best_model)
    print(f'Training completed with best loss: {best_loss:.4f} at iteration {best_it}')

    return train_loss_list

def train_phase4(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                 check_every=50, verbose=1, sparsity=100, batch_size=2048, lam_cycle=0.01, beta=1):
    '''Train model with Adam and ensure causality matrix is acyclic.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Initialize optimizer
    optimizer = optim.Adam(crvae.parameters(), lr=lr)
    
    for it in range(max_iter):
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]

        # Forward pass
        pred, mu, log_var = crvae(X_batch)
        reconstruction_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) for i in range(p)])
        
        # Calculate regularization terms
        mmd_loss = (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)).mean(dim=0)
        ridge_loss = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        smooth_loss = reconstruction_loss + ridge_loss + beta * mmd_loss
        
        # Calculate cycle loss
        adjacency_matrix = crvae.GC(threshold=False)
        cycle_loss = calculate_cycle_loss(adjacency_matrix, lam_cycle)

        # Total loss
        total_loss = smooth_loss + cycle_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        total_loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        # Log the training process
        if (it) % check_every == 0:
            with torch.no_grad():
                print(f'Iteration {it}, Total Loss: {total_loss:.4f}, '
                      f'Reconstruction Loss: {reconstruction_loss:.4f}, MMD Loss: {mmd_loss:.4f}, '
                      f'Ridge Loss: {ridge_loss:.4f}, Cycle Loss: {cycle_loss:.4f}')
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_it = it
                    best_model = deepcopy(crvae.state_dict())
    
    # Restore best model at the end
    crvae.load_state_dict(best_model)
    print(f'Training completed with best loss: {best_loss:.4f} at iteration {best_it}')

    return train_loss_list



def train_phase5(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                          check_every=50, verbose=1, sparsity=100, batch_size=1024, lam_cycle=0.01, beta=1):
    '''Train model with Adam and ensure causality matrix is acyclic using notears_linear.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    
    # Initialize optimizer
    optimizer = optim.Adam(crvae.parameters(), lr=lr)
    
    for it in range(max_iter):
        idx = np.random.randint(len(X), size=(batch_size,))
        X_batch = X[idx]

        # Forward pass
        pred, mu, log_var = crvae(X_batch)
        reconstruction_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) for i in range(p)])
        
        # Calculate regularization terms
        mmd_loss = (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=-1).sum(dim=0)).mean(dim=0)
        ridge_loss = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        smooth_loss = reconstruction_loss + ridge_loss + beta * mmd_loss
        
        # Extract current causality matrix
        current_causality_matrix = crvae.GC(threshold=False).cpu().detach().numpy()
        
        # Calculate cycle loss using notears_linear
        W_est = notears_linear(current_causality_matrix, lambda1=lam_cycle, loss_type='l2')
        cycle_loss = np.sum((W_est - current_causality_matrix)**2)
        
        # Convert cycle_loss back to tensor and to the correct device
        cycle_loss_tensor = torch.tensor(cycle_loss, dtype=torch.float32, device=device)
        
        # Total loss
        total_loss = smooth_loss + cycle_loss_tensor
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log the training process
        if (it) % check_every == 0:
            print(f'Iteration {it}, Total Loss: {total_loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                  f'MMD Loss: {mmd_loss.item():.4f}, Ridge Loss: {ridge_loss.item():.4f}, Cycle Loss: {cycle_loss:.4f}')
            train_loss_list.append(total_loss.item())

    return train_loss_list

class MatrixExp(Function):
    @staticmethod
    def forward(ctx, W):
        W_squared = W @ W
        E = torch.matrix_exp(W_squared)
        ctx.save_for_backward(E, W)
        return E

    @staticmethod
    def backward(ctx, grad_output):
        E, W = ctx.saved_tensors
        grad_input = grad_output @ E @ W * 2
        return grad_input
    
def update_crvae_gc_matrix(crvae, W_est, device='cuda:1'):
    """
    此函数更新CRVAE模型的GRU层权重以反映notears_linear_torch计算得到的因果关系矩阵。
    """
    # 将W_est转换为适当的形状和类型以匹配CRVAE网络参数
    for i, net in enumerate(crvae.networks):
        with torch.no_grad():
            # 假设我们将W_est应用到每个网络的GRU权重
            # 注意：这里的更新策略可能需要根据模型的实际结构进行调整
            shape = net.gru.weight_ih_l0.shape
            updated_weight = W_est[:, i].reshape(shape)
            net.gru.weight_ih_l0.copy_(updated_weight)

def notears_linear_torch(X, lambda1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, device='cuda:1'):
    n, d = X.shape
    X = X.to(device)
    W = torch.zeros((d, d), dtype=torch.float32, device=device, requires_grad=True)

    def closure():
        optimizer.zero_grad()
        loss = _loss(W, X, n, loss_type)
        E = MatrixExp.apply(W)
        h = torch.trace(E) - d
        penalty = lambda1 * torch.norm(W, 1)
        augmented_lagrangian = loss + penalty + 0.5 * rho * (h ** 2) + alpha * h
        augmented_lagrangian.backward()
        return augmented_lagrangian

    def _loss(W, X, n, loss_type):
        M = torch.mm(X, W)
        if loss_type == 'l2':
            R = X - M
            return 0.5 / n * torch.sum(R ** 2)
        else:
            raise ValueError('Unknown loss type')
    
    optimizer = torch.optim.LBFGS([W], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

    rho, alpha = 1.0, 0.0  # 初始化 rho 和 alpha

    for it in range(max_iter):
        optimizer.step(closure)
        with torch.no_grad():
            E = MatrixExp.apply(W)
            h = torch.trace(E) - d
            if h.item() <= h_tol or rho >= rho_max:
                break
            alpha += rho * h.item()  # 更新 alpha
            rho = min(rho * 10, rho_max)  # 更新 rho
    
    W_est = W.detach()
    W_est[torch.abs(W_est) < w_threshold] = 0
    return W_est

def train_phase_haha(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                 check_every=50, verbose=1, sparsity=100, batch_size=2048, lambda1=0.1, device='cuda:1'):
    '''Train model with Adam and periodically update the causality matrix using notears_linear_torch.'''
    optimizer = torch.optim.Adam(crvae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loss_list = []

    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0).to(device)
    Y_all = torch.cat(Y, dim=0).to(device)
    
    for it in range(max_iter):
        optimizer.zero_grad()
        idx = torch.randint(0, X_all.size(0), (batch_size,))
        X_batch = X_all[idx]
        Y_batch = Y_all[idx]

        p = X_batch.shape[-1]  # 定义特征数量

        pred, mu, log_var = crvae(X_batch)
        # 调整 Y_batch 以匹配 pred 的维度
        Y_batch_adjusted = Y_batch[:, :pred[0].shape[1], :]
        
        reconstruction_loss = sum([loss_fn(pred[i], Y_batch_adjusted[:, :, i].unsqueeze(-1)) for i in range(p)])
        mmd_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).sum(dim=0)).mean(dim=0)
        ridge_loss = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        total_loss = reconstruction_loss + mmd_loss + ridge_loss

        total_loss.backward()
        optimizer.step()

        if it % check_every == 0:
            with torch.no_grad():
                # Evaluate and print the training progress.
                print(f'Iteration {it}, Total Loss: {total_loss.item()}')
                train_loss_list.append(total_loss.item())
                X_2d = X_all.mean(dim=1) 
                
                # Optionally: Update the causality matrix with notears_linear_torch
                W_est = notears_linear_torch(X_2d, lambda1, 'l2', 100, 1e-8, 1e+16, 0.3, device)  # Assuming notears_linear_torch function is adjusted accordingly
                # Assume update_crvae_gc_matrix correctly updates the model with W_est
                update_crvae_gc_matrix(crvae, W_est, device)
    
    # Optionally restore the best model based on some criterion
    # crvae.load_state_dict(best_model)  # Make sure to define best_model based on your criteria

    return train_loss_list
