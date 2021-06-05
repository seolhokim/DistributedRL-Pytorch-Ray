from utils.environment import Environment
from utils.utils import make_transition
import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal

import ray
import numpy as np

def run_env(env, brain, traj_length = 0, get_traj = False, reward_scaling = 0.01):
    score = 0
    transition = None
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
            action = dist.sample()
            next_state, reward, done, _ = env.step(action)
        if get_traj :
            transition = make_transition(state,\
                                         action,\
                                         reward * reward_scaling,\
                                         next_state,\
                                         float(done))
            brain.put_data(transition)
            
        state = next_state
        score += reward
        
        if done:
            break
    return score, done

def compute_gradients(env_name, global_agent, brain, traj_length = 0, reward_scaling = 0.01):
    get_traj = True
    env = Environment(env_name)
    done = False
    while not done:
        weights = ray.get(global_agent.get_weights.remote())
        brain.set_weights(weights)
        
        _, done = run_env(env, brain, traj_length, get_traj, reward_scaling)
        
        grad = brain.compute_gradient() 
        yield grad

@ray.remote
def test_agent(env_name, agent, repeat):
    brain = ray.get(agent.get_brain.remote())
    score_lst = []
    for i in range(repeat):
        env = Environment(env_name)
        score, _ = run_env(env, brain)
        score_lst.append(score)
    return sum(score_lst)/repeat