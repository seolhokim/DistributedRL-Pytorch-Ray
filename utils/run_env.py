from utils.environment import Environment
from utils.utils import make_transition
import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal

import ray
import numpy as np
import time

def run_env(env, brain, traj_length = 0, get_traj = False, reward_scaling = 0.1):
    score = 0
    transition = None
    if traj_length == 0:
        traj_length = env._max_episode_steps
        
    if env.can_run :
        state = env.state
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
            action = dist.sample()
            next_state, reward, done, _ = env.step(action)
        
        if get_traj :
            transition = make_transition(np.array(state).reshape(1,-1),\
                                         np.array(action).reshape(1,-1),\
                                         np.array(reward * reward_scaling).reshape(1,-1),\
                                         np.array(next_state).reshape(1,-1),\
                                         np.array(float(done)).reshape(1,-1))
            brain.put_data(transition)
        score += reward
        if done:
            if not get_traj:
                break
            state = env.reset()
        else :
            state = next_state
    
    return score

@ray.remote
def test_agent(env_name, agent, repeat, sleep = 3):
    total_time = 0 
    while 1 :
        time.sleep(sleep)
        total_time += sleep
        brain = ray.get(agent.get_brain.remote())
        score_lst = []
        env = Environment(env_name)
        for i in range(repeat):
            score = run_env(env, brain)
            score_lst.append(score)
        print("time : ", total_time, "'s, ", repeat, " means performance : ", sum(score_lst)/repeat)
        if sleep == 0:
            return