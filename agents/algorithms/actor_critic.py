from networks.network import Actor, Critic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
class ActorCritic(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(ActorCritic, self).__init__()
        self.args = args
        self.actor = Actor(self.args['layer_num'], state_dim, action_dim,\
                           self.args['hidden_dim'], self.args['activation_function'],\
                           self.args['last_activation'],self.args['trainable_std'])
        self.critic = Critic(self.args['layer_num'], state_dim, 1, \
                             self.args['hidden_dim'], self.args['activation_function'],\
                             self.args['last_activation'])
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
    
    def compute_gradient(self, state_lst, action_lst, reward_lst, next_state_lst, done_lst):
        state = torch.tensor(state_lst, dtype = torch.float)
        action = torch.tensor(action_lst)
        reward = torch.tensor(reward_lst, dtype = torch.float).unsqueeze(-1)
        next_state = torch.tensor(next_state_lst, dtype = torch.float)
        done = torch.tensor(done_lst, dtype = torch.float).unsqueeze(-1)
        
        td = reward + (1 - done) * self.args['gamma'] * self.v(next_state)
        if self.args['advantage'] == True :
            advantage = td - self.v(state)
        else :
            advantage = td
        
        if self.args['discrete'] :
            prob = self.get_action(state)
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