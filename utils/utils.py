import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import gym
import ray
import numpy as np

class Environment:
    def __init__(self,env_name):
        self.env = gym.make(env_name)
        self._max_episode_steps = self.env._max_episode_steps
        self.can_run = False
        self.state = None
        
    def reset(self):
        self.can_run = True
        self.state = self.env.reset()
        return self.state
    
    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        self.state = next_state
        if done == True:
            self.can_run = False
        return next_state, reward, done, info
    
class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    #def __getattr__(self,val):
    #    return self[val]

def run_env(env, brain, traj_length = 0, get_traj = False, reward_scaling = 0.01):
    score = 0
    state_lst, action_lst, reward_lst, next_state_lst, done_lst = [], [], [], [], []
    
    if traj_length == 0:
        traj_length = env._max_episode_steps
        
    if env.can_run :
        state = np.array(env.state)
    else :
        state = env.reset()
        
    for t in range(traj_length):
        if brain.args['discrete'] :
            prob = brain.get_action(torch.from_numpy(state).float())
            dist = Categorical(prob)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)
        else :#continuous
            mu,std = brain.get_action(torch.from_numpy(state).float())
            dist = Normal(mu,std)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step([action])
        if get_traj :
            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append(reward * reward_scaling)
            next_state_lst.append(next_state)
            done_lst.append(done)
            
        state = next_state
        score += reward
        
        if done:
            break
    return score, done, (state_lst, action_lst, reward_lst, next_state_lst, done_lst)

#discrete
def compute_gradients(env_name, global_agent, brain, traj_length = 0, reward_scaling = 0.01):
    get_traj = True
    env = Environment(env_name)
    done = False
    while not done:
        weights = ray.get(global_agent.get_weights.remote())
        brain.set_weights(weights)
        
        _, done, traj = run_env(env, brain, traj_length, get_traj, reward_scaling)
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = traj
        
        grad = brain.compute_gradient(state_lst, action_lst, reward_lst, next_state_lst, done_lst) 
        yield grad

@ray.remote
def test_agent(env_name, agent, repeat):
    brain = ray.get(agent.get_brain.remote())
    score_lst = []
    for i in range(repeat):
        env = Environment(env_name)
        score, _, _ = run_env(env, brain)
        score_lst.append(score)
    return sum(score_lst)/repeat