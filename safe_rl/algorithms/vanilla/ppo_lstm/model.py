import numpy as np

from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal

    
class LSTMActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hid_dim=128, activation=F.tanh):
        super(LSTMActorCritic, self).__init__()
        
        if isinstance(act_space, Box):  # safety-gymnasium
            obs_dim = obs_space.shape[0]
            act_dim = act_space.shape[0]
        
        elif isinstance(act_space, Discrete):  # highway-env
            obs_dim = obs_space.shape[0]
            act_dim = act_space.n

        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.lstm = nn.LSTM(hid_dim, hid_dim)
        self.fc_pi = nn.Linear(hid_dim, act_dim)
        self.fc_v = nn.Linear(hid_dim, 1)
        self.fc_cv = nn.Linear(hid_dim, 1)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        
        self.activation = activation

    def encode(self, x):
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        return x2
    
    def lstm_forward(self, x, h_in):
        x_lstm, h_out = self.lstm(x, h_in)
        return x_lstm, h_out
    
    def heads(self, x):
        mu = self.fc_pi(x)
        std = torch.exp(self.log_std)
        v = self.fc_v(x)
        cv = self.fc_cv(x)
        return mu, std, v, cv
    
    def pi(self, x, h_in):
        T, B, _ = x.shape
        xmlp = self.encode(x.view(T*B, -1))
        xlstm, h_out = self.lstm_forward(xmlp.view(T, B, -1), h_in)
        xlstm = xlstm.view(T*B, -1)
        x = xmlp + xlstm
        mu = self.fc_pi(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std), h_out
    
    def v(self, x, h_in):
        T, B, _ = x.shape
        x_mlp = self.encode(x.view(T*B, -1))
        x_lstm, _ = self.lstm_forward(x_mlp.view(T, B, -1), h_in)
        x_lstm = x_lstm.view(T*B, -1)
        x = x_mlp + x_lstm
        v = self.fc_v(x)
        cv = self.fc_cv(x)
        return v.squeeze(-1), cv.squeeze(-1)
    
    def step(self, x, h_in):
        with torch.no_grad():
            x = x.view(1, 1, -1)
            dist, h_out = self.pi(x, h_in)
            a = dist.sample()
            logp_a = dist.log_prob(a).sum(axis=-1)
            v, cv = self.v(x, h_in)
        return a.squeeze(0), v, cv, logp_a, h_out
    
    def act(self, x, h_in):
        return self.step(x, h_in)[0]