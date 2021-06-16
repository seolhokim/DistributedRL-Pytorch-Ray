from networks.network import Actor
from agents.algorithms.base import Agent
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor

import torch
import torch.optim as optim
import torch.nn.functional as F

class DQN(Agent):
    def __init__(self, device, state_dim, action_dim, args):
        super(DQN, self).__init__(state_dim, action_dim, args)
        self.args = args
        self.device = device
        self.q_network = Actor(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        self.target_q_network = Actor(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        if self.args['discrete'] == True : 
            action_dim = 1
        if self.args['learner'] == False:
            self.data = ReplayBuffer(True, self.args['actor_memory_size'], state_dim, action_dim)
        else :
            pass
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.args['lr'])
        self.update_cycle = 1000
        self.update_num = 0
    def get_action(self,x):
        x, _ = self.q_network(x)
        return x
    
    def get_buffer_size(self):
        return self.data.data_idx
        
    def get_trajectories(self, batch_size):
        data = self.data.sample(True, batch_size)
        return data
    
    def train_network(self, data):
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        actions = actions.type(torch.int64)
        q = self.get_action(states)
        q_action = q.gather(1, actions)
        next_q_max = self.target_q_network(next_states)[0].max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.args['gamma'] * next_q_max
        loss = F.smooth_l1_loss(q_action, target.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_num += 1
        
        if self.update_num % self.update_cycle == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            