import numpy as np
import collections

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.
        
        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0 # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._beta = beta
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size
    
    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def add(self, priority : float, experience: Experience) -> None:
        """Add a new experience to memory."""
        #priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0
    
    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size
    
    def sample(self):
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]     
        weights = (self._buffer_length * sampling_probs[idxs])**(-self._beta)
        normalized_weights = weights / weights.max()
        
        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities


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

memory = PrioritizedExperienceReplayBuffer(32, 100000)

def get_td_error(agent, data, weights = None):
    if isinstance(data, Experience):
        states, actions, rewards, next_states, dones = [torch.Tensor(vs) for vs in zip(data)]
    else :
        states, actions, rewards, next_states, dones = [torch.Tensor(vs) for vs in zip(*data)]
    actions = actions.type(torch.int64).unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    q = agent.get_q(states)
    q_action = q.gather(1, actions)
    
    target = rewards + (1 - dones) * agent.args['gamma'] * agent.target_q_network(next_states)[0].max(1)[0].unsqueeze(1)

    beta = 1
    n = torch.abs(q_action - target.detach())
    cond = n < beta
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if isinstance(weights, np.ndarray):
        return torch.tensor(weights) * loss
    else :
        return loss
    
def train(agent, memory):
    idx, data, priority = memory.sample()
    loss = get_td_error(agent, data)
    agent.optimizer.zero_grad()
    loss.mean().backward()
    agent.optimizer.step()
    td_error = loss.detach().numpy()
    memory.update_priorities(idx, td_error.squeeze())
    
for n_epi in range(10000):
    score = 0
    done = False
    state = env.reset()
    while not done :
        action = agent.get_action(torch.from_numpy(state).float().to(device))
        next_state, reward, done, _ = env.step(action)
        experiment = Experience(state, action, reward * reward_scaling, next_state, done)
        td_error = get_td_error(agent, experiment)
        memory.add(td_error, experiment)
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

