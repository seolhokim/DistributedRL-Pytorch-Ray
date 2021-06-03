import ray
import torch
import gym
import time
from configparser import ConfigParser
from argparse import ArgumentParser

from agents.workers.worker import Worker
from agents.workers.learner import Learner
from agents.algorithms.actor_critic import ActorCritic
from utils.utils import Dict, run_env, test_agent

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'CartPole-v1', help = 'environment to adjust (default : CartPole-v1)')
parser.add_argument("--algo", type=str, default = 'a3c', help = 'algorithm to adjust (default : a3c)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--num_workers', type=int, default=3, help='number of workers, (default: 3)')
parser.add_argument('--test_repeat', type=int, default=10, help='test repeat for mean performance, (default: 10)')
parser.add_argument('--test_sleep', type=int, default=3, help='test sleep time when training, (default: 3)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter() 
else :
    writer = None
    
parser = ConfigParser()
parser.read('config.ini')


agent_args = Dict(parser, args.algo) 

ray.init()

env = gym.make(args.env_name)

##action_dim = env.action_space.shape[0] continuous
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

global_agent = Learner.remote()
ray.get(global_agent.init.remote(ActorCritic(writer, device, \
                                     state_dim, action_dim, agent_args), agent_args))
local_agents = [Worker.remote() for _ in range(args.num_workers)]
ray.get([agent.init.remote(ActorCritic(writer, device, state_dim, action_dim, agent_args), \
                   agent_args) for agent in local_agents])

start = time.time()
for i in range(args.num_workers):
    [agent.compute_gradients.remote(args.env_name, global_agent, args.epochs) for agent in local_agents]

for i in range(100):
    print(i,'-th test performance : ', (ray.get(test_agent.remote(args.env_name, global_agent, args.test_repeat))))
    time.sleep(args.test_sleep)
print("time :", time.time() - start)