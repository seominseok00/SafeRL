import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MLPActor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_dim, hid_dim, act_dim, activation=F.tanh, act_limit=1):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.mu_layer = nn.Linear(hid_dim, act_dim)
        self.log_std_layer = nn.Linear(hid_dim, act_dim)
        self.activation = activation
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = self.activation(self.fc1(obs))
        net_out = self.activation(self.fc2(x))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()  # Draw a sample with reparameterization for gradient backpropagation

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
    

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim, activation=F.tanh):
        super(MLPQFunction, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, obs, act):
        x = self.activation(self.fc1(torch.cat([obs, act], dim=-1)))
        x = self.activation(self.fc2(x))
        q = self.fc3(x)
        return torch.squeeze(q, -1)
    
    
class MLPActorCritic(nn.Module):
    init_alpha = 0.2

    def __init__(self, obs_space, act_space, hid_dim=64, activation=F.tanh):
        super(MLPActorCritic, self).__init__()

        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        act_limit = act_space.high[0]

        self.pi = MLPActor(obs_dim, hid_dim, act_dim, activation, act_limit)

        self.q1 = MLPQFunction(obs_dim, act_dim, hid_dim, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hid_dim, activation)

        self.qc1 = MLPQFunction(obs_dim, act_dim, hid_dim, activation)
        self.qc2 = MLPQFunction(obs_dim, act_dim, hid_dim, activation)

        self.log_alpha = torch.tensor(np.log(self.init_alpha), requires_grad=True)
        
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
        
class MLPPenalty(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=64, activation=F.tanh):
        super(MLPPenalty, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, obs, act):
        x = self.activation(self.fc1(torch.cat([obs, act], dim=-1)))
        x = self.activation(self.fc2(x))
        q = self.fc3(x)
        return torch.squeeze(q, -1)