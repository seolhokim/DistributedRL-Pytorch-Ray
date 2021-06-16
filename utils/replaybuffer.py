import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_copy, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.data = {}
        self.buffer_copy = buffer_copy
        self.data['priority'] = np.zeros((self.max_size, 1))
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        
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