from agents.algorithms.base import ActorCritic
from utils.utils import convert_to_tensor, make_mini_batch
from utils.run_env import run_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical

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

    def reset(self, env, reward_scaling = 0.1):
        get_traj = True
        run_env(env, self, self.device, self.args['traj_length'], get_traj, reward_scaling)
        data = self.data.sample(shuffle = False)
        
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        if self.args['discrete'] :
            prob = self.get_action(states)
            actions = actions.type(torch.int64)
            dist = Categorical(prob)
            entropy = dist.entropy() * self.args['entropy_coef']
            old_log_probs = torch.log(prob.gather(1, actions))
        else :
            mu,std = self.get_action(states)
            dist = torch.distributions.Normal(mu,std)
            entropy = dist.entropy() * self.args['entropy_coef']
            old_log_probs = dist.log_prob(actions).sum(1,keepdim = True)
        
        old_values, advantages = self.get_gae(states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)

        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantages = advantages
        self.returns = returns
        self.old_values = old_values

    def compute_gradients(self):
        self.zero_grad()
        for state,action,old_log_prob,advantage,return_,old_value \
        in make_mini_batch(self.args['batch_size'], self.states, self.actions, \
                                       self.old_log_probs, self.advantages, self.returns, self.old_values):
            if self.args['discrete'] :
                prob = self.get_action(state)
                action = action.type(torch.int64)
                dist = Categorical(prob)
                entropy = dist.entropy() * self.args['entropy_coef']
                log_prob = torch.log(prob.gather(1, action))
            else :
                mu,std = self.get_action(state)
                dist = torch.distributions.Normal(mu,std)
                entropy = dist.entropy() * self.args['entropy_coef']
                log_prob = dist.log_prob(action).sum(1,keepdim = True)
            value = self.v(state).float()

            #policy clipping
            ratio = torch.exp(log_prob - old_log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.args['max_clip'], 1+self.args['max_clip']) * advantage
            actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 

            #value clipping (PPO2 technic)
            old_value_clipped = old_value + (value - old_value).clamp(-self.args['max_clip'],self.args['max_clip'])
            value_loss = (value - return_.detach().float()).pow(2)
            value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
            critic_loss = 0.5 * self.args['critic_coef'] * torch.max(value_loss,value_loss_clipped).mean()
            loss = actor_loss + critic_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.args['max_grad_norm'])
        return self.get_gradients()                         