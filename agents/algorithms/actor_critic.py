from networks.network import Actor, Critic
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor, make_mini_batch
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
    
    def compute_gradients(self, env, global_agent, epochs, reward_scaling = 0.1):
        get_traj = True
        for i in range(epochs):
            weights = ray.get(global_agent.get_weights.remote())
            self.set_weights(weights)
            run_env(env, self, self.args['traj_length'], get_traj, reward_scaling)
            for grad in self.compute_gradients_() :
                yield grad
                
    def get_gae(self, states, rewards, next_states, dones):
        values = self.v(states).detach()
        td_target = rewards + self.args['gamma'] * self.v(next_states) * (1 - dones)
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if dones[idx] == 1:
                advantage = 0.0
            advantage = self.args['gamma'] * self.args['lambda_'] * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages
    
    def compute_gradients_(self):
        data = self.data.sample(shuffle = False)
        
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        
        if self.args['use_gae'] :
            old_values, advantages = self.get_gae(states, rewards, next_states, dones)
            returns = advantages + old_values
        
        else :
            returns = rewards + self.args['gamma'] * self.v(next_states) * (1 - dones)
            advantages = returns - self.v(states)
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)
        for state, action, advantage, return_ in \
        make_mini_batch(64, states, actions, advantages, returns) :
            if self.args['discrete'] :
                prob = self.get_action(state)
                action = action.type(torch.int64)
                log_prob = torch.log(prob.gather(1, action))
            else :
                mu, std = self.get_action(state)
                dist = Normal(mu,std)
                log_prob = dist.log_prob(action).sum(1,keepdim = True)
            loss = - log_prob * advantage.detach() + \
                F.smooth_l1_loss(self.v(state), return_.detach().float())
            loss = loss.mean()
            self.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), self.args['max_grad_norm'])
            yield self.get_gradients()
    
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