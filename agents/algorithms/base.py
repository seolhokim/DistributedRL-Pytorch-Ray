from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.replaybuffer import ReplayBuffer
from networks.network import Actor, Critic

class AgentBase(nn.Module,metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(AgentBase, self).__init__()
    def put_data(self, transition):
        pass
    def get_action(self, x):
        pass
    def v(self, x):
        pass
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass
    def get_gradients(self):
        pass
    def set_gradients(self, gradients):
        pass
    
class Agent(AgentBase):
    def __init__(self, state_dim, action_dim, args):
        super(Agent, self).__init__()
    def name(self):
        return self.__class__.__name__.lower()
    
    def put_data(self, transition):
        self.data.put_data(transition)
        
    def get_action(self, x):
        pass
    
    def v(self, x):
        pass
    
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
                #It makes all process slower than cpu
                p.grad = g.to(p.device)

    def add_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is None :
                pass
            elif p.grad == None :
                #p.grad = torch.zeros(g.shape)
                p.grad = torch.zeros(g.shape).to(p.device)
            if g is not None:
                #p.grad += g
                #It makes all process slower than cpu
                p.grad += g.to(p.device)
    
class ActorCritic(Agent):
    def __init__(self, state_dim, action_dim, args):
        super(ActorCritic, self).__init__(state_dim, action_dim, args)
        self.args = args
        self.actor = Actor(self.args['layer_num'], state_dim, action_dim,\
                           self.args['hidden_dim'], self.args['activation_function'],\
                           self.args['last_activation'],self.args['trainable_std'])
        self.critic = Critic(self.args['layer_num'], state_dim, 1, \
                             self.args['hidden_dim'], self.args['activation_function'],\
                             self.args['last_activation'])
        if self.args['discrete'] :
            action_dim = 1
        self.data = ReplayBuffer(max_size = self.args['traj_length'], state_dim = state_dim, num_action = action_dim)
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
    
