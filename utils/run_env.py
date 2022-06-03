from utils.environment import Environment
from utils.utils import make_transition
import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal

import ray
import numpy as np
import time

def run_env(env, algorithm, device, traj_length = 0, get_traj = False, reward_scaling = 0.1):
    score = 0
    transition = None
    if traj_length == 0:
        traj_length = env._max_episode_steps

    if env.can_run :
        state = env.state
    else :
        state = env.reset()
    for t in range(traj_length):
        if algorithm.args['value_based'] :
            if algorithm.args['discrete'] :
                action = algorithm.get_action(torch.from_numpy(state).float().to(device))
                log_prob = np.zeros((1,1))##
            else :
                pass
        else :
            if algorithm.args['discrete'] :
                prob = algorithm.get_action(torch.from_numpy(state).float().to(device))
                dist = Categorical(prob)
                action = dist.sample()
                log_prob = torch.log(prob.reshape(1,-1).gather(1, action.reshape(1,-1))).detach().cpu().numpy()
                action = action.item()
            else :#continuous
                mu,std = algorithm.get_action(torch.from_numpy(state).float().to(device))
                dist = Normal(mu,std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1,keepdim = True).detach().cpu().numpy()
                action = action.detach().cpu()
        next_state, reward, done, _ = env.step(action)
        if get_traj :
                transition = make_transition(np.array(state).reshape(1,-1),\
                                             np.array(action).reshape(1,-1),\
                                             np.array(reward * reward_scaling).reshape(1,-1),\
                                             np.array(next_state).reshape(1,-1),\
                                             np.array(float(done)).reshape(1,-1),\
                                            np.array(log_prob))
                algorithm.put_data(transition)
        score += reward
        if done:
            if not get_traj:
                break
            state = env.reset()
        else :
            state = next_state
    return score

class TestAgent:
    def __init__(self, env_name, actor, writer, device, \
                     state_dim, action_dim, agent_args, ps, repeat, sleep = 3):
        self.env_name = env_name
        self.agent = actor.algorithm
        self.ps = ps
        self.repeat = repeat
        self.device = device
        self.sleep = sleep
    def test_agent(self):
        total_time = 0 
        while 1 :
            time.sleep(self.sleep)
            total_time += self.sleep
            self.agent.set_weights(ray.get(self.ps.pull.remote()))
            score_lst = []
            env = Environment(self.env_name)
            for i in range(self.repeat):
                score = run_env(env, self.agent, self.device)
                score_lst.append(score)
            print("time : ", total_time, "'s, ", self.repeat, " means performance : ", sum(score_lst)/self.repeat)
            if self.sleep == 0:
                return