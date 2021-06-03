import ray
import torch
import gym

from agents.workers.worker import Worker
from agents.workers.learner import Learner
from agents.algorithms.actor_critic import ActorCritic
from utils.utils import Dict, run_env, test_agent
import time
from configparser import ConfigParser

# Hyperparameters
n_train_processes = 3
learning_rate = 3e-4
update_interval = 500
gamma = 0.98
max_train_ep = 1
epoch = 1000
test_cycle = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#if args.use_cuda == False:
#    device = 'cpu'

from torch.utils.tensorboard import SummaryWriter ##arg
writer = SummaryWriter() ##arg

parser = ConfigParser()
parser.read('config.ini')

agent_args = Dict(parser,'actor_critic') ##arg

ray.init()

env_name = 'CartPole-v1'##arg
env = gym.make(env_name) ##arg

##action_dim = env.action_space.shape[0] continuous
action_dim = env.action_space.n

state_dim = env.observation_space.shape[0]
global_agent = Learner.remote()
ray.get(global_agent.init.remote(ActorCritic(writer, device, \
                                     state_dim, action_dim, agent_args), agent_args))
local_agents = [Worker.remote() for _ in range(n_train_processes)]
ray.get([agent.init.remote(ActorCritic(writer, device, state_dim, action_dim, agent_args), \
                   agent_args) for agent in local_agents])


start = time.time()
for i in range(epoch * n_train_processes):
    result_ids = [agent.compute_gradients.remote(env_name, global_agent) for agent in local_agents]
    done_id, result_ids = ray.wait(result_ids)

    if i % (n_train_processes * test_cycle) == 0 :
        print(i,'-th test performance : ', ray.get(test_agent.remote(env_name, global_agent, 1)))
        time.sleep(1)
print("time :", time.time() - start)