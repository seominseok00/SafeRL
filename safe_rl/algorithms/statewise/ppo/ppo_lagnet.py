import argparse
import os
import time
import yaml
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Adam

from safe_rl.algorithms.statewise.ppo.model import MLPActorCritic, MLPLagrangeMultiplier
from safe_rl.algorithms.statewise.ppo.buffer import Buffer

from safe_rl.utils.config import load_config, get_device

def ppo_lagnet(config, actor_critic=MLPActorCritic, ac_kwargs=dict(), env_lib="safety_gymnasium", use_cost_indicator=True,
               env_id='SafetyPointGoal1-v0', seed=0, epochs=300, steps_per_epoch=30000, gamma=0.99, lamda=0.97,
               clip_ratio=0.2, target_kl=0.01, lagrange_network=MLPLagrangeMultiplier, lagrange_init=1.0, pi_lr=3e-4, vf_lr=1e-3, lagrange_lr=3e-4,
               cost_limit=25, train_pi_iters=80, train_v_iters=80, train_lagrange_iters=5, max_ep_len=1000, device=None):
    
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

    buf = Buffer(obs_space, act_space, steps_per_epoch, gamma, lamda)
    
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)

    # Create directory for saving logs and models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(current_dir, '../../../'))
    run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-') + f'ppo-lagnet-{env_id}'
    run_dir = os.path.join(root_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    #=====================================================================#
    #  Define Lagrangian multiplier network for penalty learning          #
    #=====================================================================#

    lagrange_net = lagrange_network(obs_space, **ac_kwargs, lagrange_init=lagrange_init).to(device)
    lagrange_optimizer = Adam(lagrange_net.parameters(), lr=lagrange_lr)

    #=====================================================================#
    #  Loss function for update policy                                    #
    #=====================================================================#

    def compute_loss_pi(data):
        obs, act, adv, cadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'], data['logp']

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        surr_adv = (torch.min(ratio * adv, clip_adv))

        surro_cost = (ratio * cadv)

        lagrange_multiplier = lagrange_net(obs).detach()

        pi_objective = surr_adv - lagrange_multiplier * surro_cost
        pi_objective = pi_objective / (1 + lagrange_multiplier)
        loss_pi = -pi_objective
        loss_pi = loss_pi.mean()

        approx_kl = (logp_old - logp).mean().item()
        pi_info = dict(kl=approx_kl)

        return loss_pi, pi_info
    
    #=====================================================================#
    #  Loss function for update value function                            #
    #=====================================================================#

    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']

        loss_v = ((ac.v(obs) - ret) ** 2).mean()
        loss_vc = ((ac.vc(obs) - cret) ** 2).mean()

        return loss_v, loss_vc
    
    #=====================================================================#
    #  Loss function for update Lagrange Multiplier Network               #
    #=====================================================================#

    def compute_loss_lagrange(data):
        obs, cost = data['obs'], data['crew']
        cost_dev = cost - cost_limit

        lagrange_multiplier = lagrange_net(obs)
        loss_lagrange = -lagrange_multiplier * cost_dev
        loss_lagrange = loss_lagrange.mean()

        return loss_lagrange, cost_dev
    
    def update():
        train_logger = {
            'lagrange': [],
            'cost_dev': [],
            'loss_pi': [],
            'loss_v': [],
            'loss_cv': [],
            'loss_lagrange': []
        }

        data = buf.get()
        data = {k: v.to(device) for k, v in data.items()}

        #=====================================================================#
        #  Update Lagrange Multiplier Nework                                  #
        #=====================================================================#

        for i in range(train_lagrange_iters):
            loss_lagrange, cost_dev = compute_loss_lagrange(data)

            lagrange_optimizer.zero_grad()
            loss_lagrange.backward()
            lagrange_optimizer.step()

            train_logger['lagrange'].append(lagrange_net(data['obs']).mean().item())
            train_logger['cost_dev'].append(cost_dev.mean().item())
            train_logger['loss_lagrange'].append(loss_lagrange.item())

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
            loss_v, loss_cv = compute_loss_v(data)

            vf_optimizer.zero_grad()
            loss_v.backward()
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_cv.backward()
            cvf_optimizer.step()

            train_logger['loss_v'].append(loss_v.item())
            train_logger['loss_cv'].append(loss_cv.item())

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
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))

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
            'lagrange': np.mean(train_logger['lagrange']),
            'loss_pi': np.mean(train_logger['loss_pi']),
            'loss_v': np.mean(train_logger['loss_v']),
            'loss_cv': np.mean(train_logger['loss_cv']),
            'loss_lagrange': np.mean(train_logger['loss_lagrange']),
        })

        # Save log
        epoch_logger_df = pd.DataFrame(epoch_logger)
        epoch_logger_df.to_csv(os.path.join(run_dir, 'ppo_lagnet.csv'), index=False)

        # Save model
        torch.save(ac.state_dict(), os.path.join(run_dir, 'ppo_lagnet.pth'))
        torch.save(lagrange_net.state_dict(), os.path.join(run_dir, 'lagrange_net.pth'))

        # Save best model
        current_return = np.mean(rollout_logger['EpRet'])
        current_cost = np.mean(rollout_logger['EpCost'])

        if current_return >= best_return and current_cost <= lowest_cost:
            best_return = current_return
            lowest_cost = current_cost
            torch.save(ac.state_dict(), os.path.join(run_dir, 'best_ppo_lagnet.pth'))
            torch.save(lagrange_net.state_dict(), os.path.join(run_dir, 'best_lagrange_net.pth'))

        print('Epoch: {} avg return: {}, avg cost: {}, lagrange: {}, cost_dev: {}'.format(epoch, current_return, current_cost, np.mean(train_logger['lagrange']), np.mean(train_logger['cost_dev'])))
        print('Loss pi: {}, Loss v: {}, Loss cv: {}, Loss lagrange: {}\n'.format(np.mean(train_logger['loss_pi']), np.mean(train_logger['loss_v']), np.mean(train_logger['loss_cv']), np.mean(train_logger['loss_lagrange'])))

    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statewise PPO Lagrangian Network')
    parser.add_argument('--config', type=str, default='configs/statewise/ppo/ppo_lagnet.yaml', help='Path to the YAML configuration file (relative to project root)')
    args = parser.parse_args()

    config, original_config = load_config(args.config)
    ppo_lagnet(config=original_config, **config)