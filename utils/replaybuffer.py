import numpy as np
import ray
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_copy, max_size, state_dim, num_action, n_step = 1, args = None):
        self.max_size = max_size
        self.n_step = n_step
        self.data_idx = 0
        self.data = {}
        self.buffer_copy = buffer_copy
        self.data['priority'] = np.zeros((self.max_size, 1))
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        if self.n_step > 1 :
            self.n_step_buffer = deque(maxlen=self.n_step)
            self.args = args
    def input_data(self, idx, key, transition):
        data_len = len(transition[key])
        if self.buffer_copy == True:
            data = np.copy(self.data[key])
        else :
            data = self.data[key]
        if (idx + data_len) < self.max_size :
            data[idx : idx + data_len] = transition[key]
        else :
            data[idx : self.max_size] = transition[key][: (self.max_size - idx)]
            data[ : data_len - (self.max_size - idx)] = transition[key][(self.max_size - idx) :]
        if self.buffer_copy == True:
            self.data[key] = data
        return data_len
    
    def put_data(self, transition):
        if self.n_step > 1 :
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_step :
                return
            else :
                n_step_reward = self.n_step_buffer[-1]['reward']
                for traj in reversed(list(self.n_step_buffer)[:-1]):
                    reward, done = traj['reward'], traj['done']
                    n_step_reward = reward + (1 - done) * self.args['gamma'] * n_step_reward
                traj['reward'] = n_step_reward
        idx = self.data_idx % self.max_size
        for key in transition.keys():
            data_len = self.input_data(idx, key, transition)
        self.data_idx += data_len
        
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sampled_data = {}
            sample_num = min(self.max_size, self.data_idx)
            if sample_num < batch_size:
                return sampled_data
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            for key in self.data.keys():
                sampled_data[key] = self.data[key][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)
    
@ray.remote
class CentralizedBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action):
        self.temp_buffer = deque(maxlen=learner_memory_size)
        self.buffer =  ReplayBuffer(False, learner_memory_size, state_dim, num_action)
        self.max_iter = 50
    def put_trajectories(self, data):
        self.temp_buffer.append(data)
    
    def get_temp_buffer(self):
        return self.temp_buffer
    
    def get_buffer(self):
        return self.buffer
    
    def sample(self,batch_size):
        return self.buffer.sample(shuffle = True, batch_size = batch_size)
    
    def stack_data(self):
        size = min(len(self.temp_buffer), self.max_iter)
        data = [self.temp_buffer.popleft() for _ in range(size)]
        for i in range(size):
            self.buffer.put_data(data[i])