from agents.algorithms.base import ActorCritic
from utils.utils import convert_to_tensor, make_mini_batch
from utils.run_env import run_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import ray

class A3C(ActorCritic):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(A3C, self).__init__(state_dim, action_dim, args)
        self.device = device
        self.args = args

    def compute_gradients(self, env, ps, epochs, reward_scaling = 0.1):
        get_traj = True
        for i in range(epochs):
            weights = ray.get(ps.pull.remote())
            self.set_weights(weights)
            run_env(env, self, self.device, self.args['traj_length'], get_traj, reward_scaling)
            for grad in self.compute_gradients_() :
                yield grad
                
    
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
        make_mini_batch(self.args['batch_size'], states, actions, advantages, returns) :
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