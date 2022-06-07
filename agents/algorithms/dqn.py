from networks.network import Actor
from agents.algorithms.base import Agent
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor, make_transition, Experience

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
            return torch.tensor(weights) * loss
        else :
            return loss
    def get_td_error2(self, data, weights = None):
        if isinstance(data, Experience):
            states, actions, rewards, next_states, dones, _ = [torch.Tensor(vs) for vs in zip(data)]
        else :
            states, actions, rewards, next_states, dones, _ = [torch.Tensor(vs) for vs in zip(*data)]
        actions = actions.type(torch.int64) 
        q = self.get_q(states)
        q_action = q.gather(1, actions)
        target = rewards + (1 - dones) * self.args['gamma'] * self.target_q_network(next_states)[0].max(1)[0].unsqueeze(1)
        beta = 1
        n = torch.abs(q_action - target.detach())
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
        if isinstance(weights, np.ndarray):
            return torch.tensor(weights).unsqueeze(1) * loss
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
    
    def train_network(self, sample):
        idx, data, priority = sample
        loss = self.get_td_error2(data, priority)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.update_num += 1
        if self.update_num % self.args['target_update_cycle'] == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        return idx, loss.detach().numpy()