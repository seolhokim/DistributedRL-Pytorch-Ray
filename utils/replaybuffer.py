import numpy as np

class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.action_prob_exist = action_prob_exist
        self.data = {}

        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        if self.action_prob_exist :
            self.data['log_prob'] = np.zeros((self.max_size, 1))
    
    def input_data(self, idx, key, transition):
        data = np.copy(self.data[key])
        data[idx] = transition[key]
        self.data[key] = data
        
    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        self.input_data(idx, 'state', transition)
        self.input_data(idx, 'action', transition)
        self.input_data(idx, 'reward', transition)
        self.input_data(idx, 'next_state', transition)
        self.input_data(idx, 'done', transition)
        if self.action_prob_exist :
            self.input_data(idx, 'log_prob', transition)
        
        self.data_idx += 1
        
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist :
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)