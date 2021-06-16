import ray
import time

import torch
import torch.nn.functional as F
import gym

from configparser import ConfigParser
import torch.optim as optim
import numpy as np
from utils.utils import Dict,convert_to_tensor
from utils.run_env import test_agent, run_env
from utils.environment import Environment
from utils.replaybuffer import ReplayBuffer, CentralizedBuffer
from agents.algorithms.base import Agent
from agents.algorithms.dqn import DQN
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
    
learner_memory_size = 100000

            
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
        

actor_memory_size = 5000
@ray.remote
class Actor:
    def init(self, num, brain, args):
        self.num = num
        self.brain = brain
        self.args = args
        #run_env해서 data에 일단 꽉채운뒤부터 random으로 보내는걸로 가자.
        run_env(env, self.brain, actor_memory_size, True)
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
    