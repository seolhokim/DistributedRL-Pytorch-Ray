import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
import torch.optim as optim

from utils.utils import make_transition, convert_to_tensor
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
    def get_action(self,x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)
    

from agents.algorithms.base import ActorCritic
class Impala(ActorCritic):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(Impala, self).__init__(state_dim, action_dim, args)
        self.device = device
        self.args = args
        self.actor = Actor()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.args['lr'])#
        #self.data = []
        
    def compute_gradients(self, env, epochs):
        for n_epi in range(epochs):
            s = env.reset()
            done = False
            for i in range(self.args['traj_length']):
                prob = self.actor.get_action(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, info = env.step(a.item())
                transition = make_transition(np.array(s).reshape(1,-1),\
                                                         np.array(a).reshape(1,-1),\
                                                         np.array(r).reshape(1,-1),\
                                                         np.array(s_prime).reshape(1,-1),\
                                                         np.array(float(done)).reshape(1,-1),\
                                                         np.array(torch.log(prob[a]).detach()).reshape(1,-1))
                self.put_data(transition)
                if done :
                    s = env.reset()
                else :
                    s = s_prime
            self.compute_gradients_(self.data)
            
    def get_vtrace(self, rewards, dones, v, next_v, log_probs, old_log_probs):
        rho = torch.min(torch.tensor(self.args['rho_bar']),torch.exp(log_probs - old_log_probs))
        c = torch.min(torch.tensor(self.args['c_bar']),torch.exp(log_probs - old_log_probs))
        delta_v = rho * (
            rewards + self.args['gamma'] * next_v - v) #(1 - dones) 위치
        vtrace = torch.zeros((self.args['traj_length']+1,1), device=self.device)
        vtrace[-1] = next_v[-1]
        vtrace[-2] = next_v[-1]
        for i in reversed(np.arange(self.args['traj_length']-1)):
            vtrace[i] = v[i] + (1 - dones[i]) * (delta_v[i] +  self.args['gamma'] * c[i] * (vtrace[i+1] - next_v[i]))
        return rho.detach(), vtrace.detach()
        
    def compute_gradients_(self,data):
        data = self.data.sample(shuffle = False)
        
        states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'],data['log_prob'])
        actions = actions.type(torch.int64)
        prob = self.actor.get_action(states)
        log_probs = torch.log(prob.gather(1, actions))
        v = self.v(states)
        next_v = self.v(next_states)
        next_v = next_v * (1 - dones)
        rho, v_trace = self.get_vtrace(rewards, dones, v, next_v, log_probs, old_log_probs)
        critic_loss = F.smooth_l1_loss(v, v_trace[:-1])
        actor_loss = - (rho * log_probs * (rewards + (self.args['gamma'] * \
                           v_trace[1:] * (1 - dones) - v)).detach()).mean()
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class Args:
    def __init__(self):
        self.env_name = 'CartPole-v1'#'Hopper-v2'#'CartPole-v1'#'MountainCarContinuous-v0'#"MountainCarContinuous-v0"#"Pendulum-v0"##'CartPole-v1'
        self.algo = 'impala'
        self.epochs = 1000
        self.num_actors = 3
        self.test_repeat = 10
        self.test_sleep = 3
        self.use_cuda = False
        self.tensorboard = False
args = Args()
import ray

from configparser import ConfigParser
from argparse import ArgumentParser

from run_algorithm import run_apex, run_dppo, run_a3c
from utils.utils import Dict
parser = ArgumentParser('parameters')

##Algorithm config parser
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo) 

from utils.utils import run_setting
args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)

agent = Impala(writer, device, state_dim, action_dim, agent_args)

import gym
env = gym.make(args.env_name)

def test_agent(env_name, agent, repeat):
    total_time = 0 
    score_lst = []
    env = gym.make()
    for i in range(repeat):
        score = run_env(env, agent)
        score_lst.append(score)
    print("time : ", total_time, "'s, ", repeat, " means performance : ", sum(score_lst)/repeat)
    
test_itertaion = 1
start = time.time()
for i in range(1000):
    agent.compute_gradients(env, 1)
    if i % 20 == 0 :
        score = 0.0
        for test_iter in range(test_itertaion):
            s = env.reset()
            done = False

            while not done: # CartPole-v1 forced to terminates at 500 step.
                prob = agent.actor.get_action(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, info = env.step(a.item())
                #pi.put_data((r,prob[a]))
                s = s_prime
                score += r
        print("# of episode :{}, avg score : {}".format(i, score/test_itertaion))
print(time.time() - start) 