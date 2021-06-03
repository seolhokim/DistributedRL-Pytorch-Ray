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
    
#discrete
def run_env(env_name, global_agent, brain, traj_length = 0):
    env = gym.make(env_name)
    score_lst = []
    if traj_length == 0:
        traj_length = env._max_episode_steps
    score = 0
    done = False
    state = env.reset()
    while not done:
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = [], [], [], [], []
        for t in range(traj_length):
            weights = ray.get(global_agent.get_weights.remote())
            brain.set_weights(weights)
            prob = brain.get_action(torch.from_numpy(state).float())
            dist = Categorical(prob)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)

            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append(reward/100.0)
            next_state_lst.append(next_state)
            done_lst.append(done)
            
            state = next_state

            score += reward
            if done:
                break
        grad = brain.train_network(state_lst, action_lst,reward_lst, next_state_lst, done_lst)
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
            prob = brain.get_action(torch.from_numpy(state).float())
            dist = Categorical(prob)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)

            state = next_state

            score += reward
            if done:
                break
        score_lst.append(score)
    return sum(score_lst)/repeat
                    