import numpy as np

import torch

def discount_cumsum(x, discount):
    result = np.zeros_like(x, dtype=np.float32)
    running_sum = 0
    for t in reversed(range(len(x))):
        running_sum = x[t] + discount * running_sum
        result[t] = running_sum
    return result

class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lamda=0.97):
        self.gamma = gamma
        self.lamda = lamda

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)

        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.crew_buf = np.zeros([size], dtype=np.float32)

        self.ret_buf = np.zeros([size], dtype=np.float32)
        self.cret_buf = np.zeros([size], dtype=np.float32)

        self.adv_buf = np.zeros([size], dtype=np.float32)
        self.cadv_buf = np.zeros([size], dtype=np.float32)

        self.val_buf = np.zeros([size], dtype=np.float32)
        self.cval_buf = np.zeros([size], dtype=np.float32)

        self.logp_buf = np.zeros([size], dtype=np.float32)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, crew, val, cval, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew
        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp

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
            logp=self.logp_buf
        )
        
        # Convert to torch tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}