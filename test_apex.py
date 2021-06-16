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
from agents.runners.actors.apex_actor import APEXActor
from agents.runners.learners.apex_learner import APEXLearner
class Args:
    def __init__(self):
        self.env_name = 'CartPole-v1'#'Hopper-v2'#'CartPole-v1'#'MountainCarContinuous-v0'#"MountainCarContinuous-v0"#"Pendulum-v0"##'CartPole-v1'
        self.algo = 'apex'
        self.epochs = 1000
        self.num_actors = 5
        self.test_repeat = 10
        self.test_sleep = 3
        self.use_cuda = False
        self.tensorboard = False
args = Args()

parser = ConfigParser()
parser.read('config.ini')

agent_args = Dict(parser, args.algo)
        
env = Environment(args.env_name)
state_dim = env.state_dim
action_dim = env.action_dim
agent_args['discrete'] = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ray.init()
learner = APEXLearner.remote()
agent_args['learner'] = True
learner.init.remote(DQN(device, state_dim, action_dim, agent_args), agent_args)

actors = [APEXActor.remote() for _ in range(args.num_actors)]

buffer = CentralizedBuffer.remote(agent_args['learner_memory_size'], state_dim, 1) #action_dim 
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
    