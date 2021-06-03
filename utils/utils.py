import torch
from torch.distributions import Categorical
import gym

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
def run_env(env_name, brain, train, update_interval = 0):
    env = gym.make(env_name)
    if update_interval == 0:
        update_interval = env._max_episode_steps
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
            if done:
                break
        
        if train : 
            grad = brain.train_network(state_lst, action_lst,reward_lst, next_state, done)
            yield grad
        else :
            return reward_lst
