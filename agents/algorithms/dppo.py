from agents.algorithms.base import ActorCritic
from utils.utils import convert_to_tensor, make_mini_batch
from utils.run_env import run_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import ray

class DPPO(ActorCritic):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(DPPO, self).__init__(state_dim, action_dim, args)
        self.device = device
                
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

    def compute_gradients(self, env, global_agent, epochs, reward_scaling = 0.1):
        get_traj = True
        weights = ray.get(global_agent.get_weights.remote())
        self.set_weights(weights)
        run_env(env, self, self.args['traj_length'], get_traj, reward_scaling)
        return self.compute_gradients_()
    
    def compute_gradients_(self):
        data = self.data.sample(shuffle = False)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        
        mu,sigma = self.get_action(states)
        dist = torch.distributions.Normal(mu,sigma)
        old_log_probs = dist.log_prob(actions).sum(-1,keepdim = True)
        
        old_values, advantages = self.get_gae(states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)
        
        for i in range(1):
            for state,action,old_log_prob,advantage,return_,old_value \
            in make_mini_batch(self.args['batch_size'], states, actions, \
                                           old_log_probs,advantages,returns,old_values): 
                curr_mu,curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.args['entropy_coef']
                curr_log_prob = curr_dist.log_prob(action).sum(1,keepdim = True)

                #policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.args['max_clip'], 1+self.args['max_clip']) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                #value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.args['max_clip'],self.args['max_clip'])
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.args['critic_coef'] * torch.max(value_loss,value_loss_clipped).mean()
                
                loss = actor_loss + critic_loss
                self.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.args['max_grad_norm'])
                yield self.get_gradients()
                