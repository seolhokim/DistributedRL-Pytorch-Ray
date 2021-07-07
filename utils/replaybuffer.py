import numpy as np
import ray
import random
from collections import deque

from utils.utils import make_transition
from utils.sum_tree import SumTree

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
        self.data['log_prob'] = np.zeros((self.max_size, 1))
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


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
@ray.remote
class ApexBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action):
        self.append_buffer = deque(maxlen = learner_memory_size)
        self.update_buffer = deque(maxlen = learner_memory_size)
        self.buffer = Memory(learner_memory_size)
        self.max_iter = 50
        
    def put_trajectories(self, data):
        self.append_buffer.append(data)
        
    def put_idxs(self,idxs):
        self.update_buffer.append(idxs)
        
    def get_append_buffer(self):
        return self.append_buffer
    
    def get_update_buffer(self):
        return self.update_buffer
    
    def get_buffer(self):
        return self.buffer
    
    def sample(self,batch_size):
        return self.buffer.sample(batch_size)
    
    def stack_data(self):
        size = min(len(self.append_buffer), self.max_iter)
        data = [self.append_buffer.popleft() for _ in range(size)]
        for i in range(size):
            priority, state, action, reward, next_state, done = \
            data[i]['priority'], data[i]['state'], data[i]['action'], data[i]['reward'], data[i]['next_state'], data[i]['done']
            for j in range(len(data[i])):
                self.buffer.add(priority[j].item(), [state[j], action[j], reward[j], next_state[j], done[j]])
                
    def update_idxs(self):        
        size = min(len(self.update_buffer), self.max_iter)
        data = [self.update_buffer.popleft() for _ in range(size)]
        for i in range(size):
            idxs, td_errors = data[i]
            for j in range(len(idxs)):
                self.buffer.update(idxs[j], td_errors[j].item())
@ray.remote
class ImpalaBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action, args):
        self.append_buffer = deque(maxlen = learner_memory_size)
        self.args = args
        
    def put_trajectories(self, data):
        self.append_buffer.append(data)
        
    def get_append_buffer(self):
        return self.append_buffer
    
    def sample(self):
        if len(self.append_buffer) == 0 :
            return [], 0
        size = min(len(self.append_buffer), self.args['traj_num'])
        data = [self.append_buffer.popleft() for _ in range(size)]
        state, action, reward, next_state, done, log_prob = [], [], [], [], [], [] 
        for d in (data):
            state.append(d['state'])
            action.append(d['action'])
            reward.append(d['reward'])
            next_state.append(d['next_state'])
            done.append(d['done'])
            log_prob.append(d['log_prob'])
        state, action, reward, next_state, done, log_prob = np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done), np.stack(log_prob)
        transition = make_transition(state, action, reward, next_state, done, log_prob)
        return transition, size
        