import argparse
import os
import time
import yaml
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

from safe_rl.algorithms.vanilla.ppo_lstm.model import LSTMActorCritic
from safe_rl.algorithms.vanilla.ppo_lstm.buffer import LSTMBuffer, split_total_steps_to_chunks

from safe_rl.utils.config import load_config, get_device

def ppo(config, actor_critic=LSTMActorCritic, ac_kwargs=dict(), env_lib="safety_gymnasium", use_cost_indicator=True, 
        env_id='SafetyPointGoal1-v0', seed=0, epochs=300, steps_per_epoch=30000, gamma=0.99, lamda=0.97,
        clip_ratio=0.2, target_kl=0.01, ac_lr=3e-4, train_ac_iters=80, max_ep_len=1000, device=None):
    
    device = get_device(device)

    epoch_logger = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_lib == "gymnasium":
        import gymnasium
        import highway_env
        
        env = gymnasium.make(env_id)

    elif env_lib == "safety_gym":
        import gym
        import safety_gym

        env = gym.make(env_id)
        env.constrain_indicator = use_cost_indicator

    elif env_lib == "safety_gymnasium":
        import safety_gymnasium

        env = safety_gymnasium.make(env_id)
        env.task.cost_conf.constrain_indicator = use_cost_indicator
    
    else:
        raise ValueError("Unsupported environment library: {}".format(env_lib))
    
    obs_space = env.observation_space
    act_space = env.action_space

    ac = actor_critic(obs_space, act_space, **ac_kwargs).to(device)

    buf = LSTMBuffer(obs_space, act_space, 128, steps_per_epoch, gamma, lamda)

    ac_optimizer = Adam(ac.parameters(), lr=ac_lr)

    # Create directory for saving logs and models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(current_dir, '../../../'))
    run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-') + f'ppo-{env_id}'
    run_dir = os.path.join(root_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    #=====================================================================#
    #  Loss function for update policy                                    #
    #=====================================================================#

    def compute_loss_ac(data):
        loss_dict = {}

        T, B, _ = data['obs'].shape
        s, a, logp_old, adv = data['obs'], data['act'], data['logp'], data['adv']

        a = a.view(T*B, -1)
        logp_old = logp_old.view(T*B)
        adv = adv.view(T*B)
        ret = data['ret'].view(T*B)
        cret = data['cret'].view(T*B)

        dist, _ = ac.pi(s, (data['h0_h'], data['h0_c']))
        logp_a = dist.log_prob(a).sum(axis=-1)
        ratio = torch.exp(logp_a - logp_old)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv

        v, cv = ac.v(s, (data['h0_h'], data['h0_c']))

        loss_dict['loss_pi'] = -torch.min(surr1, surr2).mean()
        loss_dict['loss_v'] = ((v - ret) ** 2).mean()
        loss_dict['loss_cv'] = ((cv - cret) ** 2).mean()

        loss = loss_dict['loss_pi'] + loss_dict['loss_v'] + loss_dict['loss_cv']

        approx_kl = (logp_old - logp_a).mean().item()
        pi_info = dict(kl=approx_kl)

        return loss, loss_dict, pi_info

    #=====================================================================#
    #  Loss function for update value function                            #
    #=====================================================================#
    
    
    def update():
        train_logger = {
            'loss_pi': [],
            'loss_v': [],
            'loss_cv': [],
        }

        data = buf.get()
        data = {k: v.to(device) for k, v in data.items()}

        chunks = split_total_steps_to_chunks(data, episode_steps=1000, chunk_T=100, drop_last_incomplete=True)

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#

        for i in range(train_ac_iters):
            for data in chunks:
                loss, loss_dict, pi_info = compute_loss_ac(data)
                kl = pi_info['kl']

                # Early Stopping
                if kl > 1.5 * target_kl:
                    break

                ac_optimizer.zero_grad()
                loss.backward()
                ac_optimizer.step()

                train_logger['loss_pi'].append(loss_dict['loss_pi'].item())
                train_logger['loss_v'].append(loss_dict['loss_v'].item())
                train_logger['loss_cv'].append(loss_dict['loss_cv'].item())

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

    best_return, lowest_cost = -np.inf, np.inf
    
    h_out = (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))
    if env_lib == "gymnasium":
        o, info = env.reset()
    elif env_lib == "safety_gym":
        o = env.reset()
    elif env_lib == "safety_gymnasium":
        o, _ = env.reset()
    ep_ret, ep_cret, ep_len = 0, 0, 0

    print("🚀 Training on device: ", {next(ac.parameters()).device})

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            h_in = h_out
            a, v, vc, logp, h_out = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device), h_in)

            if env_lib == "gymnasium":
                next_o, r, d, truncated, info = env.step(a)
                c = info['cost']
            elif env_lib == "safety_gym":
                next_o, r, d, info = env.step(a)
                c = info.get('cost', 0)
            elif env_lib == "safety_gymnasium":
                next_o, r, c, d, truncated, info = env.step(a)
            
            ep_ret += r
            ep_cret += c
            ep_len += 1

            buf.store(o, a, r, c, v, vc, logp, h_in[0], h_in[1])

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off due to end of epoch')
                
                if timeout or epoch_ended:
                    _, v, vc, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device), h_out)
                else:
                    v, vc = 0, 0

                buf.finish_path(last_val=v, last_cval=vc)
                
                if terminal:
                    rollout_logger['EpRet'].append(ep_ret)
                    rollout_logger['EpCost'].append(ep_cret)
                    rollout_logger['EpLen'].append(ep_len)

                h_out = (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))
                if env_lib == "gymnasium":
                    o, info = env.reset()
                elif env_lib == "safety_gym":
                    o = env.reset()
                elif env_lib == "safety_gymnasium":
                    o, _ = env.reset()
                ep_ret, ep_cret, ep_len = 0, 0, 0

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
        
        # Save log
        epoch_logger_df = pd.DataFrame(epoch_logger)
        epoch_logger_df.to_csv(os.path.join(run_dir, 'ppo.csv'), index=False)

        # Save model
        torch.save(ac.state_dict(), os.path.join(run_dir, 'ppo.pth'))

        # Save best model
        current_return = np.mean(rollout_logger['EpRet'])
        current_cost = np.mean(rollout_logger['EpCost'])

        if current_return >= best_return and current_cost <= lowest_cost:
            best_return = current_return
            lowest_cost = current_cost
            torch.save(ac.state_dict(), os.path.join(run_dir, 'best_ppo.pth'))

        print('Epoch: {} avg return: {}, avg cost: {}'.format(epoch, current_return, current_cost))
        print('Loss pi: {}, Loss v: {}\n'.format(np.mean(train_logger['loss_pi']), np.mean(train_logger['loss_v'])))

    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--config', type=str, default='configs/vanilla/ppo_lstm/ppo.yaml', help='Path to the YAML configuration file (relative to project root)')
    args = parser.parse_args()

    config, original_config = load_config(args.config)
    ppo(config=original_config, **config)