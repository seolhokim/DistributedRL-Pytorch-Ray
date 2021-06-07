from networks.network import Actor, Critic
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor
from utils.run_env import run_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import ray

class ActorCritic(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(ActorCritic, self).__init__()
        self.device = device
        self.args = args
        self.actor = Actor(self.args['layer_num'], state_dim, action_dim,\
                           self.args['hidden_dim'], self.args['activation_function'],\
                           self.args['last_activation'],self.args['trainable_std'])
        self.critic = Critic(self.args['layer_num'], state_dim, 1, \
                             self.args['hidden_dim'], self.args['activation_function'],\
                             self.args['last_activation'])
        if self.args['discrete'] :
            self.data = ReplayBuffer(action_prob_exist = False, max_size = self.args['traj_length'], state_dim = state_dim, num_action = 1)
        else :
            self.data = ReplayBuffer(action_prob_exist = False, max_size = self.args['traj_length'], state_dim = state_dim, num_action = action_dim)
        
    def put_data(self, transition):
        self.data.put_data(transition)
        
    def get_action(self, x):
        if self.args['discrete'] :
            mu,_ = self.actor(x)
            prob = F.softmax(mu, -1)
            return prob
        else :
            mu,std = self.actor(x)
            return mu, std
        
    def v(self, x):
        return self.critic(x)
    
    def compute_gradients(self, env, global_agent, traj_length = 0, reward_scaling = 0.1):
        get_traj = True
        for i in range(traj_length):
            weights = ray.get(global_agent.get_weights.remote())
            self.set_weights(weights)

            run_env(env, self, traj_length, get_traj, reward_scaling)
            grad = self.compute_gradients_() 
            yield grad
            
    def compute_gradients_(self):
        data = self.data.sample(shuffle = False)
        
        state, action, reward, next_state, done = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        print(state.shape)
        td = reward + (1 - done) * self.args['gamma'] * self.v(next_state)
        if self.args['advantage'] == True :
            advantage = td - self.v(state)
        else :
            advantage = self.v(state)
        
        if self.args['discrete'] :
            prob = self.get_action(state)
            action = action.type(torch.int64)
            log_prob = torch.log(prob.gather(1, action))
        else :
            mu, std = self.get_action(state)
            dist = Normal(mu,std)
            log_prob = dist.log_prob(action).sum(1,keepdim = True)
        loss = - log_prob * advantage.detach() + \
            F.smooth_l1_loss(self.v(state), td.detach().float())
        loss = loss.mean()
        self.zero_grad()
        loss.backward()

        return self.get_gradients()
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g