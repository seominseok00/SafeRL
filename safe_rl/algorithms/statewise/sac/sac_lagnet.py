import argparse
import os
import time
import yaml
import itertools
from copy import deepcopy
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

from safe_rl.algorithms.vanilla.sac.model import MLPActorCritic, MLPLagrangeMultiplier
from safe_rl.algorithms.vanilla.sac.buffer import Buffer

from safe_rl.utils.config import load_config

def sac_lag(config, actor_critic=MLPActorCritic, ac_kwargs=dict(), env_lib="safety_gymnasium", env_id='SafetyPointGoal1-v0',
            use_cost_indicator=True, seed=0, epochs=300, steps_per_epoch=4000, max_ep_len=1000, replay_size=int(1e6), batch_size=100,
            gamma=0.99, polyak=0.995, lagrange_network=MLPLagrangeMultiplier, lagrange_kwargs=dict(), lagrange_init=0.0, pi_lr=1e-3, q_lr=1e-3, alpha_lr=1e-3, lagrange_lr=1e-5, cost_limit=25,
            start_steps=10000, warmup_epochs= 100, update_after=1000, update_interval=50, lagrange_update_interval=25, update_iters=50, num_test_episodes=10):
    
    epoch_logger = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_lib == "gymnasium":
        import gymnasium
        import highway_env

        env, test_env = gymnasium.make(env_id), gymnasium.make(env_id)

    elif env_lib == "safety_gymnasium":
        import safety_gymnasium

        env, test_env = safety_gymnasium.make(env_id), safety_gymnasium.make(env_id)
        
        env.task.cost_conf.constrain_indicator = use_cost_indicator
        test_env.task.cost_conf.constrain_indicator = use_cost_indicator

    else:
        raise ValueError("Unsupported environment library: {}".format(env_lib))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
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

    if ac_kwargs['auto_alpha']:
        alpha_optimizer = Adam([ac.log_alpha], lr=alpha_lr)

    target_entropy = -float(act_dim)

    # Create directory for saving logs and models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.normpath(os.path.join(current_dir, '../../../'))
    run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-') + f'sac-lagnet-{env_id}'
    run_dir = os.path.join(root_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    #=====================================================================#
    #  Define Lagrangian multiplier network for penalty learning          #
    #=====================================================================#

    lagrange_net = lagrange_network(obs_dim, act_dim, **lagrange_kwargs, lagrange_init=lagrange_init)
    lagrange_optimizer = Adam(lagrange_net.parameters(), lr=lagrange_lr)


    #=====================================================================#
    #  Loss function for update policy                                    #
    #=====================================================================#

    def compute_loss_pi(data):
        o = data['obs']

        a, logp_a = ac.pi(o)

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        q = torch.min(q1, q2)

        # Entropy regularized policy loss
        reward_term = (ac.log_alpha.exp() * logp_a - q)  # alpha * logp_a - Q(s, a)
        '''
        Q(s, a)는 최대화시키고, logp_a는 최소화시킴(logp_a를 최소화하려면 정책이 무작위적이어야 하고 -> 따라서 alpha * logp_a를 최소화하는 것은 엔트로피를 키우는 방향으로 업데이트 하는 것)
        SAC는 Q(s, a)가 높은 행동을 많이 하도록 학습하는데, 그러면 local optima에 빠질 수 있음. => 엔트로피 항을 추가하여 균형있게 학습
        => 이렇게 학습할 경우 아래 두 가지를 균형있게 학습 (alpha가 이를 조절하는 entropy regularization coefficient)
        1. Q(s, a): 현재 정책이 선택한 행동이 얼마나 좋은지
        2. alpha * logp_a: 현재 정책이 얼마나 무작위적인지 (엔트로피)
        '''

        qc1 = ac.qc1(o, a)
        qc2 = ac.qc2(o, a)
        qc = torch.min(qc1, qc2)

        lagrange_multiplier = lagrange_net(o).detach()

        cost_term = lagrange_multiplier * qc

        pi_objective = reward_term + cost_term
        pi_objective = pi_objective / (1 + lagrange_multiplier)
        loss_pi = pi_objective.mean()

        if ac_kwargs['auto_alpha']:
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

            '''
            SAC는 entropy를 보상의 일부로 간주해서 Q function으로 나타내려 함
            p_a2는 확률이어서 0~1 사이의 값인데, 이 값에 log를 취하면 음수가되니까, 이를 보상으로 처리하기 위해 -ac.log_alpha.exp() * logp_a2를 더해줌 (정책이 무작위일수록 보상이 더 커지도록)
            '''
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
    
    def compute_loss_lagrange(data):
        o, a, cost = data['obs'], data['act'], data['crew']
        
        with torch.no_grad():
            qc1 = ac.qc1(o, a)
            qc2 = ac.qc2(o, a)
            qc = torch.min(qc1, qc2)

        cost_dev = qc - cost_limit

        lagrange_multiplier = lagrange_net(o)
        
        loss_lagrange = -lagrange_multiplier * cost_dev
        loss_lagrange = loss_lagrange.mean()

        return loss_lagrange

    def update(data, lagrange_update=False):
        train_logger = {
            'alpha': [],
            'lagrange': [],
            'loss_pi': [],
            'loss_q': [],
            'loss_qc': [],
            'loss_alpha': [],
            'loss_lagrange': []
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

        loss_pi, loss_alpha = compute_loss_pi(data)

        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        if ac_kwargs['auto_alpha']:
            alpha_optimizer.zero_grad()
            loss_alpha.backward()
            alpha_optimizer.step()

        train_logger['alpha'].append(ac.log_alpha.exp().item())
        train_logger['loss_pi'].append(loss_pi.item())
        train_logger['loss_alpha'].append(loss_alpha.item())

        #=====================================================================#
        #  Update Lagrange multiplier network                                 #
        #=====================================================================#
        if lagrange_update:
            loss_lagrange = compute_loss_lagrange(data)

            lagrange_optimizer.zero_grad()
            loss_lagrange.backward()
            lagrange_optimizer.step()
        else:
            loss_lagrange = torch.tensor(0.0)

        train_logger['lagrange'].append(lagrange_net(data['obs']).mean().item())
        train_logger['loss_lagrange'].append(loss_lagrange.item())

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
            if env_lib == "gymnasium":
                o, info = test_env.reset()
            elif env_lib == "safety_gymnasium":
                o, _ = test_env.reset()
            d = False
            ep_ret, ep_cret, ep_len = 0, 0, 0

            while not (d or ep_len == max_ep_len):
                if env_lib == "gymnasium":
                    o, r, d, truncated, info = test_env.step(a)
                    c = info['cost']
                elif env_lib == "safety_gymnasium":
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

    best_return, lowest_cost = -np.inf, np.inf

    update_logger = defaultdict(list)

    if env_lib == "gymnasium":
        o, info = env.reset()
    elif env_lib == "safety_gymnasium":
        o, _ = env.reset()
    ep_ret, ep_cret, ep_len = 0, 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            total_steps += 1

            if total_steps > start_steps:
                a = get_action(o)
            else:
                a = env.action_space.sample()

            if env_lib == "gymnasium":
                next_o, r, d, truncated, info = env.step(a)
                c = info['cost']
            elif env_lib == "safety_gymnasium":
                next_o, r, c, d, truncated, info = env.step(get_action(o, True))

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

                if env_lib == "gymnasium":
                    o, info = env.reset()
                elif env_lib == "safety_gymnasium":
                    o, _ = env.reset()
                ep_ret, ep_cret, ep_len = 0, 0, 0

            #=====================================================================#
            #  Run RL update                                                      #
            #=====================================================================#
            
            if total_steps >= update_after and total_steps % update_interval == 0:
                for j in range(update_iters):
                    batch = buf.sample_batch(batch_size)

                    if epoch > warmup_epochs and j % lagrange_update_interval == 0:
                        train_logger = update(batch, True)
                    else:
                        train_logger = update(batch, False)

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
            'lagrange': np.mean(update_logger['lagrange']),
            'loss_pi': np.mean(update_logger['loss_pi']),
            'loss_q': np.mean(update_logger['loss_q']),
            'loss_qc': np.mean(update_logger['loss_qc']),
            'loss_alpha': np.mean(update_logger['loss_alpha']),
            'loss_lagrange': np.mean(update_logger['loss_lagrange']),
        })

        # Save log
        epoch_logger_df = pd.DataFrame(epoch_logger)
        epoch_logger_df.to_csv(os.path.join(run_dir, 'sac_lagnet.csv'), index=False)

        # Save model
        torch.save(ac.state_dict(), os.path.join(run_dir, 'sac_lagnet.pth'))

        # Save best model
        current_return = np.mean(test_logger['TestEpRet'])
        current_cost = np.mean(test_logger['TestEpCost'])

        if current_return >= best_return and current_cost <= lowest_cost:
            best_return = current_return
            lowest_cost = current_cost
            torch.save(ac.state_dict(), os.path.join(run_dir, 'best_sac_lagnet.pth'))

        print('Epoch: {} avg return: {}, avg cost: {}, alpha: {}, lagrange: {}'.format(epoch, np.mean(rollout_logger['EpRet']), np.mean(rollout_logger['EpCost']), np.mean(update_logger['alpha']), np.mean(update_logger['lagrange'])))
        print('Test avg return: {}, avg cost: {}'.format(current_return, current_cost))
        print('Loss pi: {}, Loss q: {}, Loss qc: {}, Loss alpha: {}, Loss lagrange: {}\n'.format(np.mean(update_logger['loss_pi']), np.mean(update_logger['loss_q']), np.mean(update_logger['loss_qc']), np.mean(update_logger['loss_alpha']), np.mean(update_logger['loss_lagrange'])))

        update_logger.clear()
        
    end_time = time.time()
    print('Training time: {}h {}m {}s'.format(int((end_time - start_time) // 3600), int((end_time - start_time) % 3600 // 60), int((end_time - start_time) % 60)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC Lagrangian Network')
    parser.add_argument('--config', type=str, default='configs/statewise/sac/sac_lagnet.yaml', help='Path to the YAML configuration file (relative to project root)')
    args = parser.parse_args()

    config, original_config = load_config(args.config)
    sac_lag(config=original_config, **config)