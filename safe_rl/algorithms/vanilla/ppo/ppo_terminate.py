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

from safe_rl.algorithms.vanilla.ppo.model import MLPActorCritic
from safe_rl.algorithms.vanilla.ppo.buffer import Buffer

from safe_rl.utils.config import load_config, get_device

def ppo_terminate(config, actor_critic=MLPActorCritic, ac_kwargs=dict(), env_lib="safety_gymnasium", use_cost_indicator=True, 
                  env_id='SafetyPointGoal1-v0', seed=0, epochs=300, steps_per_epoch=30000, gamma=0.99, lamda=0.97, clip_ratio=0.2,
                  target_kl=0.01, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, max_ep_len=1000, penalty_weight=1 ,device=None):
    
    device = get_device(device)

    epoch_logger = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_lib == "gymnasium":
        import gymnasium
        import highway_env
        
        env = gymnasium.make(env_id)

    elif env_lib == "safety_gymnasium":
        import safety_gymnasium

        env = safety_gymnasium.make(env_id)
        env.task.cost_conf.constrain_indicator = use_cost_indicator
    
    else:
        raise ValueError("Unsupported environment library: {}".format(env_lib))
    
    obs_space = env.observation_space
    act_space = env.action_space

    ac = actor_critic(obs_space, act_space, **ac_kwargs).to(device)

    buf = Buffer(obs_space, act_space, steps_per_epoch, gamma, lamda)
    
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Create directory for saving logs and models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(current_dir, '../../../'))
    run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-') + f'ppo-terminate-{env_id}'
    run_dir = os.path.join(root_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

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
            'loss_v': []
        }

        data = buf.get()
        data = {k: v.to(device) for k, v in data.items()}

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
        'EpViolateRet': deque(maxlen=episode_per_epoch),
        'EpCost': deque(maxlen=episode_per_epoch),
        'EpLen': deque(maxlen=episode_per_epoch),
    }

    best_return, lowest_cost = -np.inf, np.inf
    
    if env_lib == "gymnasium":
        o, info = env.reset()
    elif env_lib == "safety_gymnasium":
        o, _ = env.reset()
    ep_ret, ep_cret, ep_len = 0, 0, 0

    print("🚀 Training on device: ", {next(ac.parameters()).device})

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))

            if env_lib == "gymnasium":
                next_o, r, d, truncated, info = env.step(a)
                c = info['cost']
            elif env_lib == "safety_gymnasium":
                next_o, r, c, d, truncated, info = env.step(a)

            # penalize cost with weight
            ep_ret += r - penalty_weight * c
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
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v, vc = 0, 0

                buf.finish_path(last_val=v, last_cval=vc)
                
                if terminal:
                    rollout_logger['EpRet'].append(ep_ret)
                    rollout_logger['EpCost'].append(ep_cret)
                    rollout_logger['EpLen'].append(ep_len)

                if env_lib == "gymnasium":
                    o, info = env.reset()
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
            'EpViolateRet': np.mean(rollout_logger['EpViolateRet']),
            'EpCost': np.mean(rollout_logger['EpCost']),
            'EpLen': np.mean(rollout_logger['EpLen']),
            'loss_pi': np.mean(train_logger['loss_pi']),
            'loss_v': np.mean(train_logger['loss_v']),
        })
        
        # Save log
        epoch_logger_df = pd.DataFrame(epoch_logger)
        epoch_logger_df.to_csv(os.path.join(run_dir, 'ppo_terminate.csv'), index=False)

        # Save model
        torch.save(ac.state_dict(), os.path.join(run_dir, 'ppo_terminate.pth'))

        # Save best model
        current_return = np.mean(rollout_logger['EpRet'])
        current_cost = np.mean(rollout_logger['EpCost'])

        if current_return >= best_return and current_cost <= lowest_cost:
            best_return = current_return
            lowest_cost = current_cost
            torch.save(ac.state_dict(), os.path.join(run_dir, 'best_ppo_terminate.pth'))

        print('Epoch: {} avg return: {}, avg cost: {}'.format(epoch, current_return, current_cost))
        print('Loss pi: {}, Loss v: {}\n'.format(np.mean(train_logger['loss_pi']), np.mean(train_logger['loss_v'])))

    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO terminate')
    parser.add_argument('--config', type=str, default='configs/vanilla/ppo/ppo_terminate.yaml', help='Path to the YAML configuration file (relative to project root)')
    args = parser.parse_args()

    config, original_config = load_config(args.config)
    ppo_terminate(config=original_config, **config)