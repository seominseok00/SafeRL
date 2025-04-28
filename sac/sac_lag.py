import time
import itertools
from copy import deepcopy
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Adam

import safety_gymnasium

from model import MLPActorCritic
from buffer import Buffer

def sac_lag(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        epochs=300, steps_per_epoch=4000, replay_size=int(1e6), batch_size=100,
        gamma=0.99, polyak=0.995, penalty_init=1.0, pi_lr=3e-4, q_lr=1e-3, alpha_lr=1e-3, penalty_lr=1e-5, auto_alpha=False,
        warmup_epochs=20, start_steps=10000, update_after=1000, update_interval=50, update_iters=50, max_ep_len=1000, num_test_episodes=10):
    
    epoch_logger = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac_kwargs['auto_alpha'] = auto_alpha
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    buf = Buffer(obs_dim, act_dim, replay_size)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    qc_params = itertools.chain(ac.qc1.parameters(), ac.qc2.parameters())

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)
    qc_optimizer = Adam(qc_params, lr=q_lr)
    if auto_alpha:
        alpha_optimizer = Adam([ac.log_alpha], lr=alpha_lr)

    target_entropy = -float(act_dim)
    
    #=====================================================================#
    #  Define Lagrangian multiplier for penalty learning                  #
    #=====================================================================#

    penalty_param = torch.nn.Parameter(torch.tensor(penalty_init, dtype=torch.float32), requires_grad=True)
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)

    #=====================================================================#
    #  Loss function for update policy                                    #
    #=====================================================================#

    def compute_loss_pi(data, penalty_param):
        o = data['obs']

        a, logp_a = ac.pi(o)

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        q = torch.min(q1, q2)

        # Entropy regularized policy loss
        reward_term = (ac.log_alpha.exp() * logp_a - q)

        qc1 = ac.qc1(o, a)
        qc2 = ac.qc2(o, a)
        qc = torch.min(qc1, qc2)

        penalty = F.softplus(penalty_param)
        # penalty = penalty_param
        penalty_item = penalty.item()

        cost_term = penalty_item * qc

        pi_objective = reward_term + cost_term
        pi_objective = pi_objective / (1 + penalty_item)
        loss_pi = pi_objective.mean()

        if auto_alpha:
            loss_alpha = -(ac.log_alpha.exp() * (logp_a + target_entropy).detach()).mean()
        else:
            loss_alpha = torch.tensor(0.0)

        return loss_pi, loss_alpha
    
    #=====================================================================#
    #  Loss function for update q function                                #
    #=====================================================================#

    def compute_loss_q(data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['crew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from current policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_target = ac_targ.q1(o2, a2)
            q2_target = ac_targ.q2(o2, a2)
            q_target = torch.min(q1_target, q2_target)

            backup_q = r + gamma * (1 - d) * (q_target - ac.log_alpha.exp() * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup_q) ** 2).mean()
        loss_q2 = ((q2 - backup_q) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        qc1 = ac.qc1(o, a)
        qc2 = ac.qc2(o, a)

        with torch.no_grad():
            qc1_target = ac_targ.qc1(o2, a2)
            qc2_target = ac_targ.qc2(o2, a2)
            qc_target = torch.min(qc1_target, qc2_target)

            backup_qc = c + gamma * (1 - d) * qc_target

        # MSE loss against Bellman backup
        loss_qc1 = ((qc1 - backup_qc) ** 2).mean()
        loss_qc2 = ((qc2 - backup_qc) ** 2).mean()
        loss_qc = loss_qc1 + loss_qc2

        return loss_q, loss_qc
    
    def compute_loss_penalty(data, penalty_param):
        cost_limit = 25
        o, a = data['obs'], data['act']

        qc1 = ac.qc1(o, a)
        qc2 = ac.qc2(o, a)
        qc = torch.min(qc1, qc2)
        
        loss_penalty = -penalty_param * (qc - cost_limit).detach()
        loss_penalty = loss_penalty.mean()

        return loss_penalty

    def update(data):
        train_logger = {
            'alpha': [],
            'penalty': [],
            'loss_pi': [],
            'loss_q': [],
            'loss_qc': [],
            'loss_alpha': [],
            'loss_penalty': []
        }

        #=====================================================================#
        #  Update q functions                                                 #
        #=====================================================================#

        loss_q, loss_qc = compute_loss_q(data)

        q_optimizer.zero_grad()
        loss_q.backward()
        q_optimizer.step()

        qc_optimizer.zero_grad()
        loss_qc.backward()
        qc_optimizer.step()

        train_logger['loss_q'].append(loss_q.item())
        train_logger['loss_qc'].append(loss_qc.item())

        # Freeze Q-networks for computational efficiency
        for p in q_params:
            p.requires_grad = False
        for p in qc_params:
            p.requires_grad = False

        #=====================================================================#
        #  Update policy function and alpha                                   #
        #=====================================================================#

        loss_pi, loss_alpha = compute_loss_pi(data, penalty_param)

        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        if auto_alpha:
            alpha_optimizer.zero_grad()
            loss_alpha.backward()
            alpha_optimizer.step()
        
        train_logger['alpha'].append(ac.log_alpha.exp().item())
        train_logger['loss_pi'].append(loss_pi.item())
        train_logger['loss_alpha'].append(loss_alpha.item())
        
        #=====================================================================#
        #  Update penalty                                                     #
        #=====================================================================#
        loss_penalty = compute_loss_penalty(data, penalty_param)

        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        penalty_optimizer.step()
        penalty_param.data.clamp_(0.0, None)

        train_logger['penalty'].append(penalty_param.item())
        train_logger['loss_penalty'].append(loss_penalty.item())

        # Unfreeze Q-networks
        for p in q_params:
            p.requires_grad = True
        for p in qc_params:
            p.requires_grad = True

        #=====================================================================#
        #  Update target networks                                             #
        #=====================================================================#
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return train_logger

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
    
    #=====================================================================#
    #  Run test episodes using deterministic policy                       #
    #=====================================================================#

    def test_agent():
        test_logger = {
            'TestEpRet': [],
            'TestEpCost': [],
            'TestEpLen': []
        }

        for j in range(num_test_episodes):
            o, _ = test_env.reset()
            d = False
            ep_ret, ep_cret, ep_len = 0, 0, 0

            while not (d or ep_len == max_ep_len):
                o, r, c, d, truncated, info = test_env.step(get_action(o, True))
                
                ep_ret += r
                ep_cret += c
                ep_len += 1
            
            test_logger['TestEpRet'].append(ep_ret)
            test_logger['TestEpCost'].append(ep_cret)
            test_logger['TestEpLen'].append(ep_len)

        return test_logger

    
    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#
    
    start_time = time.time()

    total_steps = 0

    episode_per_epoch = steps_per_epoch // max_ep_len
    rollout_logger = {
        'EpRet': deque(maxlen=episode_per_epoch),
        'EpCost': deque(maxlen=episode_per_epoch),
        'EpLen': deque(maxlen=episode_per_epoch),
    }

    update_logger = defaultdict(list)

    o, _ = env.reset()
    ep_ret, ep_cret, ep_len = 0, 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            total_steps += 1

            if total_steps > start_steps:
                a = get_action(o)
            else:
                a = env.action_space.sample()

            next_o, r, c, d, truncated, info = env.step(a)

            ep_ret += r
            ep_cret += c
            ep_len += 1

            buf.store(o, a, r, c, next_o, d)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                rollout_logger['EpRet'].append(ep_ret)
                rollout_logger['EpCost'].append(ep_cret)
                rollout_logger['EpLen'].append(ep_len)

                o, _ = env.reset()
                ep_ret, ep_cret, ep_len = 0, 0, 0

            #=====================================================================#
            #  Run RL update                                                      #
            #=====================================================================#
            
            if total_steps >= update_after and total_steps % update_interval == 0:
                for j in range(update_iters):
                    batch = buf.sample_batch(batch_size)
                    train_logger = update(batch)

                    for k, v in train_logger.items():
                        update_logger[k] += v

        #=====================================================================#
        #  Test current policy                                                #
        #=====================================================================#

        test_logger = test_agent()

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#
        
        epoch_logger.append({
            'epoch': epoch,
            'EpRet': np.mean(rollout_logger['EpRet']),
            'EpCost': np.mean(rollout_logger['EpCost']),
            'EpLen': np.mean(rollout_logger['EpLen']),
            'TestEpRet': np.mean(test_logger['TestEpRet']),
            'TestEpCost': np.mean(test_logger['TestEpCost']),
            'TestEpLen': np.mean(test_logger['TestEpLen']),
            'alpha': np.mean(update_logger['alpha']),
            'penalty': np.mean(update_logger['penalty']),
            'loss_pi': np.mean(update_logger['loss_pi']),
            'loss_q': np.mean(update_logger['loss_q']),
            'loss_qc': np.mean(update_logger['loss_qc']),
            'loss_alpha': np.mean(update_logger['loss_alpha']),
            'loss_penalty': np.mean(update_logger['loss_penalty']),
        })

        print('Epoch: {} avg return: {}, avg cost: {}, alpha: {}, penalty: {}'.format(epoch, np.mean(rollout_logger['EpRet']), np.mean(rollout_logger['EpCost']), np.mean(update_logger['alpha']), np.mean(update_logger['penalty'])))
        print('Test avg return: {}, avg cost: {}'.format(np.mean(test_logger['TestEpRet']), np.mean(test_logger['TestEpCost'])))
        print('Loss pi: {}, Loss q: {}, Loss qc: {}, Loss alpha: {}, Loss penalty: {}\n'.format(np.mean(update_logger['loss_pi']), np.mean(update_logger['loss_q']), np.mean(update_logger['loss_qc']), np.mean(update_logger['loss_alpha']), np.mean(update_logger['loss_penalty'])))

        update_logger.clear()
        
    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))
    
    epoch_logger_df = pd.DataFrame(epoch_logger)
    epoch_logger_df.to_csv('sac_lag.csv', index=False)

    # Save model
    torch.save(ac.state_dict(), '../trained_models/sac/sac_lag.pth')

if __name__ == '__main__':
    sac_lag(lambda: safety_gymnasium.make('SafetyPointButton1-v0'))