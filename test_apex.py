import ray
import time

import torch
import torch.nn.functional as F
import gym

from collections import deque

from configparser import ConfigParser
import torch.optim as optim
import numpy as np
from utils.utils import Dict,convert_to_tensor
from utils.environment import Environment
from utils.replaybuffer import ReplayBuffer
from agents.algorithms.base import Agent


class Args:
    def __init__(self):
        self.env_name = 'CartPole-v1'#'Hopper-v2'#'CartPole-v1'#'MountainCarContinuous-v0'#"MountainCarContinuous-v0"#"Pendulum-v0"##'CartPole-v1'
        self.algo = 'apex'
        self.epochs = 1000
        self.num_workers = 5
        self.test_repeat = 10
        self.test_sleep = 3
        self.use_cuda = False
        self.tensorboard = False
args = Args()

parser = ConfigParser()
parser.read('config.ini')

agent_args = Dict(parser, args.algo)

from utils.utils import make_transition
def run_env(env, brain, traj_length = 0, get_traj = False, random = False, reward_scaling = 0.1):
    score = 0
    transition = None
    if traj_length == 0:
        traj_length = env._max_episode_steps
        
    if env.can_run :
        state = env.state
    else :
        state = env.reset()
    
    for t in range(traj_length):
        '''
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
        '''
        if brain.args['discrete'] :
            if random :
                action = env.env.action_space.sample()
            else :
                action = brain.get_action(torch.from_numpy(state).float()).argmax().item()
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
    
learner_memory_size = 100000
@ray.remote
class CentralizedBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action):
        self.temp_buffer = deque(maxlen=learner_memory_size)
        self.buffer =  ReplayBuffer(False, learner_memory_size, state_dim, num_action)
        self.max_iter = 50
    def put_trajectories(self, data):
        self.temp_buffer.append(data)
    
    def get_temp_buffer(self):
        return self.temp_buffer
    
    def get_buffer(self):
        return self.buffer
    
    def sample(self,batch_size):
        return self.buffer.sample(shuffle = True, batch_size = batch_size)
    
    def stack_data(self):
        size = min(len(self.temp_buffer), self.max_iter)
        data = [self.temp_buffer.popleft() for _ in range(size)]
        for i in range(size):
            self.buffer.put_data(data[i])
            
@ray.remote
class Learner:
    def init(self, brain, args):
        self.brain = brain
        self.args = args
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def run(self,buffer):
        batch_size = 512
        #while 1 :
        print('learner start')
        for i in range(500):
            data = ray.get(buffer.sample.remote(batch_size))
            if len(data) > 0:
                self.brain.train_network(data)
            else :
                time.sleep(0.1)
        print('learner finish')
        
from networks.network import Actor as act
class DQN(Agent):
    def __init__(self, device, state_dim, action_dim, args):
        super(DQN, self).__init__(state_dim, action_dim, args) #쓸모없는 replaybuffer있음
        self.args = args
        self.device = device
        self.actor = act(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        self.target_actor = act(self.args['layer_num'], state_dim, action_dim,\
                   self.args['hidden_dim'], self.args['activation_function'],\
                   self.args['last_activation'],self.args['trainable_std'])
        self.target_actor.load_state_dict(self.actor.state_dict())
        if self.args['discrete'] == True : 
            action_dim = 1
        if self.args['learner'] == True:
            self.data = ReplayBuffer(self.args['buffer_copy'], learner_memory_size, state_dim, action_dim)
        else :
            pass
        self.optimizer = optim.Adam(self.actor.parameters(), lr = self.args['lr'])
        self.update_cycle = 1000
        self.update_num = 0
    def get_action(self,x):
        x, _ = self.actor(x)
        return x
    
    def get_buffer_size(self):
        return self.data.data_idx
        
    def get_trajectories(self, batch_size):
        data = self.data.sample(True, batch_size)
        return data
    
    def train_network(self, data):
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
        actions = actions.type(torch.int64)
        q = self.get_action(states)
        q_action = q.gather(1, actions)
        next_q_max = self.target_actor(next_states)[0].max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.args['gamma'] * next_q_max
        loss = F.smooth_l1_loss(q_action, target.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_num += 1
        
        if self.update_num % self.update_cycle == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            
actor_memory_size = 5000
@ray.remote
class Actor:
    def init(self, num, brain, args):
        self.num = num
        self.brain = brain
        self.args = args
        #run_env해서 data에 일단 꽉채운뒤부터 random으로 보내는걸로 가자.
        run_env(env, self.brain, actor_memory_size, True, True)
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def run(self, env_name, global_agent, global_buffer, epochs):
        env = Environment(env_name)
        batch_size = 16
        update_cycle = 1000
        print("actor start")
        weights = ray.get(global_agent.get_weights.remote())
        self.brain.set_weights(weights)        
        i = 0
        while 1:
        #for j in range(5000):
            run_env(env, self.brain, 1, True)
            data = self.brain.get_trajectories(batch_size)
            global_buffer.put_trajectories.remote(data)
            
            if i % update_cycle == 0:
                weights = ray.get(global_agent.get_weights.remote())
                self.brain.set_weights(weights)
            i += 1
        print('actor finish')
        
        
env = Environment(args.env_name)
state_dim = env.state_dim
action_dim = env.action_dim
agent_args['discrete'] = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ray.init()
learner = Learner.remote()
agent_args['learner'] = True
learner.init.remote(DQN(device, state_dim, action_dim, agent_args), agent_args)

actors = [Actor.remote() for _ in range(args.num_workers)]

buffer = CentralizedBuffer.remote(learner_memory_size, state_dim, 1) #action_dim 
agent_args['learner'] = False
ray.get([agent.init.remote(idx, DQN(device, state_dim, action_dim, agent_args), agent_args) for idx, agent in enumerate(actors)])

@ray.remote
def buffer_run(buffer):
    print('buffer_start')
    while 1:
    #for i in range(500):
        ray.get(buffer.stack_data.remote())
        time.sleep(0.1)
        #buffer_t = ray.get(buffer.get_buffer.remote())
    print("buffer finished")
    
[actor.run.remote(args.env_name, learner, buffer, args.epochs) for actor in actors]
test_agent.remote(args.env_name, learner, args.test_repeat, args.test_sleep)
buffer_run.remote(buffer)

time.sleep(1)
while 1 :
    learner.run.remote(buffer)
    #brain = ray.get(learner.get_brain.remote())
    #print(next(brain.actor.parameters())[0])
    time.sleep(3)
    