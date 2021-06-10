import ray
import torch
import gym
import time
from configparser import ConfigParser
from argparse import ArgumentParser

from agents.workers.worker import Worker
from agents.workers.learner import Learner
from agents.algorithms.a3c import A3C
from agents.algorithms.dppo import DPPO
from utils.utils import Dict
from utils.environment import Environment
from utils.run_env import test_agent

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'MountainCarContinuous-v0', help = 'environment to adjust (default : CartPole-v1)')
parser.add_argument("--algo", type=str, default = 'a3c', help = 'algorithm to adjust (default : a3c)')
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs, (default: 5000)')
parser.add_argument('--num_workers', type=int, default=3, help='number of workers, (default: 3)')
parser.add_argument('--test_repeat', type=int, default=10, help='test repeat for mean performance, (default: 10)')
parser.add_argument('--test_sleep', type=int, default=1, help='test sleep time when training, (default: 1)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
args = parser.parse_args()

##device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'

##tensorboard
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter() 
else :
    writer = None

##Algorithm config parser
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo) 

#ray init
ray.init()

#environment init for make agent
env = Environment(args.env_name)

state_dim = env.state_dim
action_dim = env.action_dim
agent_args['discrete'] = env.is_discrete

#agent init

if args.algo == 'a3c':
    algo = A3C
elif args.algo == 'dppo':
    algo = DPPO
global_agent = Learner.remote()
ray.get(global_agent.init.remote(algo(writer, device, \
                                     state_dim, action_dim, agent_args), agent_args))
local_agents = [Worker.remote() for _ in range(args.num_workers)]
ray.get([agent.init.remote(algo(writer, device, state_dim, action_dim, agent_args), \
                   agent_args) for agent in local_agents])

#train
start = time.time()
if agent_args['asynchronous'] :
    runners = [agent.train_agent.remote(args.env_name, global_agent, args.epochs) for agent in local_agents]
    test_agent.remote(args.env_name, global_agent, args.test_repeat, args.test_sleep)
    while len(runners) :
        done, runners = ray.wait(runners)
else :
    for i in range(args.epochs):
        runners = [agent.train_agent.remote(args.env_name, global_agent, args.epochs) for agent in local_agents]
        ray.wait(runners)
        global_agent.step_gradients.remote()
        test_agent.remote(args.env_name, global_agent, args.test_repeat, 0)
        
print("time :", time.time() - start)

#ray terminate
ray.shutdown()