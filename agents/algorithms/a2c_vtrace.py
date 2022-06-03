from agents.algorithms.base import ActorCritic
from utils.utils import convert_to_tensor, make_mini_batch
from utils.run_env import run_env
import ray
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class A2CVtrace(ActorCritic):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(A2CVtrace, self).__init__(state_dim, action_dim, args)
        self.device = device
        self.args = args
        
    def get_trajectories(self):
        data = self.data.sample(False)
        return data
    
    def compute_gradients(self, data):
        states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])

        for i in range(self.args['train_epoch']):
            if self.args['discrete'] :
                probs = self.get_action(states)
                actions = actions.type(torch.int64)
                log_probs = torch.log(probs.gather(2, actions))
            else :
                mu, std = self.get_action(states)
                dist = Normal(mu,std)
                log_probs = dist.log_prob(actions).sum(-1,keepdim = True)
            v = self.v(states)
            next_v = self.v(next_states)
            rho, v_trace = self.get_vtrace(rewards, dones, v, next_v, log_probs, old_log_probs)
            critic_loss = F.smooth_l1_loss(v, v_trace)
            actor_loss = - (rho * log_probs * (rewards + self.args['gamma'] * \
                               v_trace * (1 - dones) - v).detach()).mean()
            loss = actor_loss + critic_loss
            
            return loss 
    
    def get_vtrace(self, rewards, dones, v, next_v, log_probs, old_log_probs):
        size = rewards.shape[0]
        rho = torch.min(torch.tensor(self.args['rho_bar']),torch.exp(log_probs - old_log_probs))
        c = torch.min(torch.tensor(self.args['c_bar']),torch.exp(log_probs - old_log_probs))
        delta_v = rho * (
            rewards + self.args['gamma'] * next_v * (1 - dones) - v)
        vtrace = torch.zeros((size, self.args['traj_length']+1,1), device=self.device)

        T = v.shape[1]
        v_out = []
        v_out.append(v[:, -1] + delta_v[:, -1])
        for t in range(T - 2, -1, -1):
            _v_out = (
                v[:, t] + delta_v[:, t] + self.args['gamma'] * c[:, t] * (v_out[-1] - v[:, t + 1])
            )
            v_out.append(_v_out)
        v_out = torch.stack(list(reversed(v_out)), 1)
        return rho.detach(), v_out.detach()
