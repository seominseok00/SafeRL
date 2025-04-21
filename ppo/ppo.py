import time
from collections import deque

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

import safety_gymnasium

from model import MLPActorCritic
from buffer import Buffer

def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        epochs=300, steps_per_epoch=30000, gamma=0.99, lamda=0.97, clip_ratio=0.2,
        target_kl=0.01, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
        max_ep_len=1000):
    
    epoch_logger = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)

    buf = Buffer(obs_dim, act_dim, steps_per_epoch, gamma, lamda)
    
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    #=====================================================================#
    #  Loss function for update policy                                    #
    #=====================================================================#

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        pi_info = dict(kl=approx_kl)

        return loss_pi, pi_info
    
    #=====================================================================#
    #  Loss function for update value function                            #
    #=====================================================================#
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']

        loss_v = ((ac.v(obs) - ret) ** 2).mean()

        return loss_v
    
    def update():
        train_logger = {
            'loss_pi': [],
            'loss_v': [],
        }

        data = buf.get()

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#

        for i in range(train_pi_iters):
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']

            # Early Stopping
            if kl > 1.5 * target_kl:
                break

            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

            train_logger['loss_pi'].append(loss_pi.item())

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#

        for i in range(train_v_iters):
            loss_v = compute_loss_v(data)

            vf_optimizer.zero_grad()
            loss_v.backward()
            vf_optimizer.step()

            train_logger['loss_v'].append(loss_v.item())

        return train_logger
    

    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    
    episode_per_epoch = steps_per_epoch // max_ep_len
    rollout_logger = {
        'EpRet': deque(maxlen=episode_per_epoch),
        'EpCost': deque(maxlen=episode_per_epoch),
        'EpLen': deque(maxlen=episode_per_epoch),
    }
    
    o, _ = env.reset()
    ep_ret, ep_cret, ep_len = 0, 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, c, d, truncated, info = env.step(a)
            
            ep_ret += r
            ep_cret += c
            ep_len += 1

            buf.store(o, a, r, c, v, vc, logp)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off due to end of epoch')
                
                if timeout or epoch_ended:
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v, vc = 0, 0

                buf.finish_path(last_val=v, last_cval=vc)
                
                if terminal:
                    rollout_logger['EpRet'].append(ep_ret)
                    rollout_logger['EpCost'].append(ep_cret)
                    rollout_logger['EpLen'].append(ep_len)

                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

        #=====================================================================#
        #  Run RL update                                                      #
        #=====================================================================#

        train_logger = update()

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#

        epoch_logger.append({
            'epoch': epoch,
            'EpRet': np.mean(rollout_logger['EpRet']),
            'EpCost': np.mean(rollout_logger['EpCost']),
            'EpLen': np.mean(rollout_logger['EpLen']),
            'loss_pi': np.mean(train_logger['loss_pi']),
            'loss_v': np.mean(train_logger['loss_v']),
        })

        print('Epoch: {} avg return: {}, avg cost: {}, avg len: {}'.format(epoch, np.mean(rollout_logger['EpRet']), np.mean(rollout_logger['EpCost']), np.mean(rollout_logger['EpLen'])))
        print('Loss pi: {}, Loss v: {}\n'.format(np.mean(train_logger['loss_pi']), np.mean(train_logger['loss_v'])))

    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(
        int((end_time - start_time) // 3600),
        int((end_time - start_time) % 3600 // 60),
        int((end_time - start_time) % 60)
    ))

    epoch_logger_df = pd.DataFrame(epoch_logger)
    epoch_logger_df.to_csv('ppo.csv', index=False)

if __name__ == '__main__':
    ppo(lambda: safety_gymnasium.make('SafetyPointButton1-v0'))