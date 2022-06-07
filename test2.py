import ray

from configparser import ConfigParser
from argparse import ArgumentParser

from run_algorithm import  run_apex, run_dppo, run_a3c, run_impala
from utils.utils import Dict, boolean_string

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'CartPole-v1', help = 'environment to adjust (default : CartPole-v1)')
parser.add_argument("--algo", type=str, default = 'a3c', help = 'algorithm to adjust (default : a3c)')
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs, (default: 5000)')
parser.add_argument('--num_actors', type=int, default=3, help='number of actors, (default: 3)')
parser.add_argument('--test_repeat', type=int, default=10, help='test repeat for mean performance, (default: 10)')
parser.add_argument('--test_sleep', type=int, default=3, help='test sleep time when training, (default: 3)')
parser.add_argument("--cuda", type=boolean_string, default = True, help = 'cuda usage(default : True)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
args = parser.parse_args()

##Algorithm config parser
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo) 


from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import TestAgent
from utils.replaybuffer import ApexBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.dqn import DQN
from utils.utils import make_transition, convert_to_tensor
import torch
import torch.nn.functional as F
import numpy as np

algorithm = DQN
args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
epsilon = 0.1
agent = algorithm(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)

reward_scaling = 0.1
optimizer = torch.optim.Adam(agent.parameters(),lr = 0.001)
update_num = 0
target_update_cycle = 100
gamma = 0.98
from utils.replaybuffer import Memory
memory = Memory(100000)
def train(agent, memory):
    data = memory.sample(32)
    mini_batch, idxs, is_weights = data
    #print(mini_batch)
    import pandas as pd
    print(mini_batch)
    mini_batch = (pd.DataFrame(mini_batch))

    state = np.vstack(mini_batch.state.values)
    action = np.vstack(mini_batch.action.values)
    reward = np.vstack(mini_batch.reward.values)
    next_state = np.vstack(mini_batch.next_state.values)
    done_ = np.vstack(mini_batch.done.values)
    log_prob = np.zeros((1,1)) ###
    data = make_transition(state, action, reward, next_state, done_, log_prob)

    loss = agent.get_td_error(data)
    agent.optimizer.zero_grad()
    loss.mean().backward()
    agent.optimizer.step()
    td_error = loss.detach().numpy()
    [memory.update(idxs[i], td_error[i]) for i in range(len(idxs))]
for n_epi in range(10000):
    score = 0
    done = False
    state = env.reset()
    while not done :
        action = agent.get_action(torch.from_numpy(state).float().to(device))
        log_prob = np.zeros((1,1))
        next_state, reward, done, _ = env.step(action)
        transition = make_transition(np.array(state).reshape(1,-1),\
                                     np.array(action).reshape(1,-1),\
                                     np.array(reward * reward_scaling).reshape(1,-1),\
                                     np.array(next_state).reshape(1,-1),\
                                     np.array(float(done)).reshape(1,-1),\
                                    np.array(log_prob))
        td_error = agent.get_td_error(transition).detach().numpy()
        transition['priority'] = td_error
        memory.add(td_error, transition)
        score += reward
        if n_epi > 40:
            train(agent, memory)
            if update_num % target_update_cycle == 0:
                agent.target_q_network.load_state_dict(agent.q_network.state_dict())
            update_num += 1
        if done:
            break
        else :
            state = next_state
    if n_epi % 10 == 0 :
        print("score : ",score)

