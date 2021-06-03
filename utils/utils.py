import torch
from torch.distributions import Categorical
import gym
import ray

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
'''
#discrete
def run_env(env_name, brain, train, repeat, update_interval = 0):
    def train_network(brain, state_lst, action_lst,reward_lst, next_state, done):
        grad = brain.train_network(state_lst, action_lst,reward_lst, next_state, done)
        yield grad
    env = gym.make(env_name)
    score_lst = []
    if update_interval == 0:
        update_interval = env._max_episode_steps
    for i in range(repeat):
        score = 0
        done = False
        state = env.reset()
        while not done:
            state_lst, action_lst, reward_lst = [], [], []
            for t in range(update_interval):
                prob = brain.get_action(torch.from_numpy(state).float())
                dist = Categorical(prob)
                action = dist.sample().item()
                next_state, reward, done, _ = env.step(action)

                state_lst.append(state)
                action_lst.append([action])
                reward_lst.append(reward/100.0)
                state = next_state
                
                score += reward
                if done:
                    break
            if train : 
                return train_network(brain, state_lst, action_lst,reward_lst, next_state, done)
        score_lst.append(score)
    return sum(score_lst)/repeat
'''
def train_network(brain, state_lst, action_lst,reward_lst, next_state, done):
    grad = brain.train_network(state_lst, action_lst,reward_lst, next_state, done)
    yield grad
#discrete
def run_env(env_name, brain,  repeat, update_interval = 0):
    env = gym.make(env_name)
    score_lst = []
    if update_interval == 0:
        update_interval = env._max_episode_steps
    for i in range(repeat):
        score = 0
        done = False
        state = env.reset()
        while not done:
            state_lst, action_lst, reward_lst = [], [], []
            for t in range(update_interval):
                prob = brain.get_action(torch.from_numpy(state).float())
                dist = Categorical(prob)
                action = dist.sample().item()
                next_state, reward, done, _ = env.step(action)

                state_lst.append(state)
                action_lst.append([action])
                reward_lst.append(reward/100.0)
                state = next_state
                
                score += reward
                if done:
                    break
            grad = brain.train_network(state_lst, action_lst,reward_lst, next_state, done)
            yield grad

@ray.remote
def test_agent(env_name, agent, repeat):
    brain = ray.get(agent.get_brain.remote())
    env = gym.make(env_name)
    score_lst = []
    for i in range(repeat):
        score = 0
        done = False
        state = env.reset()
        while not done:
            state_lst, action_lst, reward_lst = [], [], []
            prob = brain.get_action(torch.from_numpy(state).float())
            dist = Categorical(prob)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)

            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append(reward/100.0)
            state = next_state

            score += reward
            if done:
                break
        score_lst.append(score)
    return sum(score_lst)/repeat
                    