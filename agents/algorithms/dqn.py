from networks.network import Actor
from agents.algorithms.base import Agent
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor, make_transition

import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np

class DQN(Agent):
    def __init__(self, writer, device, state_dim, action_dim, args, epsilon):
        super(DQN, self).__init__(state_dim, action_dim, args)
        self.args = args
        self.device = device
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.q_network = Actor(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        self.target_q_network = Actor(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.args['lr'])
        
        if self.args['discrete'] :
            action_dim = 1
        self.data = ReplayBuffer(max_size = (self.args['traj_length'] - self.args['n_step'] + 1), \
                                         state_dim = state_dim, num_action = action_dim, \
                                         n_step = self.args['n_step'], args = self.args)
        self.update_num = 0
        
    def get_q(self,x):
        x, _ = self.q_network(x)
        return x
    
    def get_td_error(self, data, weights = False):
        state, action, reward, next_state, done = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        action = action.type(torch.int64)
        q = self.get_q(state)
        q_action = q.gather(1, action)
        target = reward + (1 - done) * self.args['gamma'] * self.target_q_network(next_state)[0].max(1)[0].unsqueeze(1)

        beta = 1
        n = torch.abs(q_action - target.detach())
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

        if isinstance(weights, np.ndarray):
            return torch.tensor(weights).to(self.device) * loss
        else :
            return loss
    def get_action(self,x):
        if random.random() < self.epsilon :
            x = random.randint(0, self.action_dim - 1)
        else:
            x = self.get_q(x).argmax().item()
        return x
    
    def get_buffer_size(self):
        return self.data.data_idx
        
    def get_trajectories(self):
        data = self.data.sample(False)
        return data
    
    def train_network(self, data):
        mini_batch, idxs, is_weights = data
        mini_batch = np.array(mini_batch, dtype = object).transpose()
        state = np.vstack(mini_batch[0])
        action = np.vstack(mini_batch[1])
        reward = np.vstack(mini_batch[2])
        next_state = np.vstack(mini_batch[3])
        done = np.vstack(mini_batch[4])
        log_prob = np.zeros((1,1)) ###

        data = make_transition(state, action, reward, next_state, done, log_prob)
        td_error = self.get_td_error(data,is_weights.reshape(-1,1))
        self.optimizer.zero_grad()
        td_error.mean().backward()
        self.optimizer.step()
        self.update_num += 1
        
        if self.update_num % self.args['target_update_cycle'] == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        return idxs, td_error.detach().cpu().numpy()