import numpy as np

from gymnasium.spaces import Box, Discrete

import torch

def discount_cumsum(x, discount):
    result = np.zeros_like(x, dtype=np.float32)
    running_sum = 0
    for t in reversed(range(len(x))):
        running_sum = x[t] + discount * running_sum
        result[t] = running_sum
    return result

def split_total_steps_to_chunks(tensor_dict, episode_steps: int, chunk_T: int, drop_last_incomplete: bool = True, device=None):
    # 1. Calcualte number of episodes(E) and number of chunks (num_chunks), usable_T
    Ns = [v.shape[0] for v in tensor_dict.values()]
    N = Ns[0]  # N: total steps (batch size: episodes * episode_steps)
    assert all(n == N for n in Ns), "All tensors must share N as first dim."
    assert N % episode_steps == 0, f"N({N}) must be divisible by episode_steps({episode_steps})."
    E = N // episode_steps  # E: number of episodes

    if episode_steps % chunk_T != 0:
        if drop_last_incomplete:
            """
            Example: episode_steps=1000, chunk_T=128
            1000/128=7.8125 → 7 chunks of 128 steps
            usable_T=7*128=896
            Remaining 104 steps are discarded.
            """
            num_chunks = episode_steps // chunk_T
            usable_T = num_chunks * chunk_T
        else:
            raise ValueError("episode_steps not divisible by chunk_T.")
    else:
        """
        Example: episode_steps=1000, chunk_T=100
        1000/100=10 → 10 chunks of 100 steps
        usable_T=1000
        No steps are discarded.
        """
        num_chunks = episode_steps // chunk_T
        usable_T = episode_steps

    # 2. Except LSTM hidden state, reshape all tensors to (E, num_chunks, chunk_T, feat)
    arranged = {}
    keys_feat = []

    for k, v in tensor_dict.items():
        if k in ('hin_h', 'hin_c'):  # Except LSTM hidden states
            continue
        x = v if device is None else v.to(device)
        if x.ndim == 1:
            x = x.unsqueeze(-1)  # (N,) -> (N,1) for consistency
        F = x.shape[-1]  # feature dimension
        x = x[:E * episode_steps]  # x: (E * episode_steps, F)
        x = x.view(E, episode_steps, F)  # x: (E, episode_steps, F)
        x = x[:, :usable_T]  # truncate to usable_T
        x = x.view(E, num_chunks, chunk_T, F)  # x: (E, num_chunks, chunk_T, F)
        arranged[k] = x
        keys_feat.append(k)

    # 3. Prepare initial hidden states h0 for each chunk
    H = tensor_dict['ac_hin_h'].shape[-1]  # tensor_dict['hin_h']: (N, H)

    # Actor-Critic LSTM hidden states
    ac_hin_h = (tensor_dict['ac_hin_h'] if device is None else tensor_dict['ac_hin_h'].to(device))[:E*episode_steps]  # ac_hin_h: (E * episode_steps, H)
    ac_hin_c = (tensor_dict['ac_hin_c'] if device is None else tensor_dict['ac_hin_c'].to(device))[:E*episode_steps]
    ac_hin_h = ac_hin_h.view(E, episode_steps, H)[:, :usable_T]   # ac_hin_h: (E, episode_steps, H) -> (E, usable_T, H)
    ac_hin_c = ac_hin_c.view(E, episode_steps, H)[:, :usable_T]
    # Each chunk's initial hidden state is the hidden state at the chunk's start time
    starts = torch.arange(start=0, end=usable_T, step=chunk_T, device=ac_hin_h.device)  # starts: (C, )
    h0_h_all = ac_hin_h.index_select(dim=1, index=starts)  # h0_h_all: (E, num_chunks, H)
    h0_c_all = ac_hin_c.index_select(dim=1, index=starts)

    # Lagrange Multiplier LSTM hidden states
    l_hin_h = (tensor_dict['l_hin_h'] if device is None else tensor_dict['l_hin_h'].to(device))[:E*episode_steps]  # l_hin_h: (E * episode_steps, H)
    l_hin_c = (tensor_dict['l_hin_c'] if device is None else tensor_dict['l_hin_c'].to(device))[:E*episode_steps]
    l_hin_h = l_hin_h.view(E, episode_steps, H)[:, :usable_T, :]   # l_hin_h: (E, episode_steps, H)
    l_hin_c = l_hin_c.view(E, episode_steps, H)[:, :usable_T, :]
    # Each chunk's initial hidden state is the hidden state at the chunk's start time
    l0_h_all = l_hin_h.index_select(dim=1, index=starts)  # l0_h_all: (E, num_chunks, H)
    l0_c_all = l_hin_c.index_select(dim=1, index=starts)

    # 4. Return each chunk as (T, B, F) + h0
    out = []
    for c in range(num_chunks):
        batch_c = {}

        for k in keys_feat:
            x = arranged[k][:, c]                 # x: (E, T, F)
            x = x.permute(1, 0, 2).contiguous()   # x: (T, E=B, F)
            batch_c[k] = x

        # h0: (num_layers=1, B, H)
        h0_h = h0_h_all[:, c, :]  # (E, H)
        h0_c = h0_c_all[:, c, :]  # (E, H)
        batch_c['ac_h0_h'] = h0_h.unsqueeze(0)  # (1, E=B, H)
        batch_c['ac_h0_c'] = h0_c.unsqueeze(0)  # (1, E=B, H)

        l0_h = l0_h_all[:, c, :]  # (E, H)
        l0_c = l0_c_all[:, c, :]  # (E, H
        batch_c['l_h0_h'] = l0_h.unsqueeze(0)  # (1, E=B, H)
        batch_c['l_h0_c'] = l0_c.unsqueeze(0)  # (1, E=B, H)

        out.append(batch_c)

    return out

class LSTMBuffer:
    def __init__(self, obs_space, act_space, hid_dim, size, gamma=0.99, lamda=0.97):
        obs_dim = obs_space.shape[0]
        
        if isinstance(act_space, Box):
            is_continuous = True
            act_dim = act_space.shape[0]
        
        elif isinstance(act_space, Discrete):
            is_continuous = False
            act_dim = act_space.n

        self.gamma = gamma
        self.lamda = lamda

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32) if is_continuous else np.zeros([size], dtype=np.float32)

        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.crew_buf = np.zeros([size], dtype=np.float32)

        self.ret_buf = np.zeros([size], dtype=np.float32)
        self.cret_buf = np.zeros([size], dtype=np.float32)

        self.adv_buf = np.zeros([size], dtype=np.float32)
        self.cadv_buf = np.zeros([size], dtype=np.float32)

        self.val_buf = np.zeros([size], dtype=np.float32)
        self.cval_buf = np.zeros([size], dtype=np.float32)

        self.logp_buf = np.zeros([size], dtype=np.float32)

        self.ac_hin_h_buf = np.zeros([size, hid_dim], dtype=np.float32)
        self.ac_hin_c_buf = np.zeros([size, hid_dim], dtype=np.float32)

        self.l_hin_h_buf = np.zeros([size, hid_dim], dtype=np.float32)
        self.l_hin_c_buf = np.zeros([size, hid_dim], dtype=np.float32)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, crew, val, cval, logp, h, c, lh, lc):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew
        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        self.ac_hin_h_buf[self.ptr] = h.squeeze(0).squeeze(0).cpu().numpy()
        self.ac_hin_c_buf[self.ptr] = c.squeeze(0).squeeze(0).cpu().numpy()
        self.l_hin_h_buf[self.ptr] = lh.squeeze(0).squeeze(0).cpu().numpy()
        self.l_hin_c_buf[self.ptr] = lc.squeeze(0).squeeze(0).cpu().numpy()
        self.ptr += 1


    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)  # not include self.ptr (self.path_start_idx ~ self.ptr-1)
    
        # timeout or epoch_ended 실행시 network로 예측한 rews, crews를 저장
        # np.append() is return new array
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lamda)
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.gamma * self.lamda)

        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        self.ptr, self.path_start_idx = 0, 0

        # Advantage normalization
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() + 1e-8
        cadv_mean = self.cadv_buf.mean()

        norm_adv_buf = (self.adv_buf - adv_mean) / adv_std
        norm_cadv_buf = self.cadv_buf - cadv_mean

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            rew=self.rew_buf,
            crew=self.crew_buf,
            ret=self.ret_buf,
            cret=self.cret_buf,
            adv=norm_adv_buf,
            cadv=norm_cadv_buf,
            logp=self.logp_buf,
            ac_hin_h=self.ac_hin_h_buf,
            ac_hin_c=self.ac_hin_c_buf,
            l_hin_h=self.l_hin_h_buf,
            l_hin_c=self.l_hin_c_buf,
        )
        
        # Convert to torch tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}