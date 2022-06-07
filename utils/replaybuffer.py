import numpy as np
import ray
import random
from collections import deque

from utils.utils import make_transition, Experience

class ReplayBuffer():
    def __init__(self, max_size, state_dim, num_action, n_step = 1, args = None):
        self.max_size = max_size
        self.n_step = n_step
        self.data_idx = 0
        self.data = {}
        self.data['priority'] = np.zeros((self.max_size, 1))
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        self.data['log_prob'] = np.zeros((self.max_size, 1))
        if self.n_step > 1 :
            self.n_step_buffer = deque(maxlen=self.n_step)
            self.args = args
    def input_data(self, idx, key, transition):
        data_len = len(transition[key])
        data = self.data[key]
        if (idx + data_len) < self.max_size :
            data[idx : idx + data_len] = transition[key]
        else :
            data[idx : self.max_size] = transition[key][: (self.max_size - idx)]
            data[ : data_len - (self.max_size - idx)] = transition[key][(self.max_size - idx) :]
        self.data[key] = data
        return data_len
    
    def put_data(self, transition):
        if self.n_step > 1 :
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_step :
                return
            else :
                n_step_reward = self.n_step_buffer[-1]['reward']
                for traj in reversed(list(self.n_step_buffer)[:-1]):
                    reward, done = traj['reward'], traj['done']
                    n_step_reward = reward + (1 - done) * self.args['gamma'] * n_step_reward
                traj['reward'] = n_step_reward
        idx = self.data_idx % self.max_size
        for key in transition.keys():
            data_len = self.input_data(idx, key, transition)
        self.data_idx += data_len
        
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sampled_data = {}
            sample_num = min(self.max_size, self.data_idx)
            if sample_num < batch_size:
                return sampled_data
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            for key in self.data.keys():
                sampled_data[key] = self.data[key][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)

class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 buffer_size: int,
                 batch_size: int,
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

@ray.remote
class ApexBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action):
        self.append_buffer = deque(maxlen = learner_memory_size)
        self.update_buffer = deque(maxlen = learner_memory_size)
        self.buffer = PrioritizedExperienceReplayBuffer(learner_memory_size, 512) #fix batch_size
        self.max_iter = 50
        
    def put_trajectories(self, data):
        self.append_buffer.append(data)
        
    def put_idxs(self,idxs):
        self.update_buffer.append(idxs)
        
    def get_append_buffer(self):
        return self.append_buffer
    
    def get_update_buffer(self):
        return self.update_buffer
    
    def get_buffer(self):
        return self.buffer
    
    def sample(self,batch_size):
        return self.buffer.sample()
    
    def stack_data(self):
        size = min(len(self.append_buffer), self.max_iter)
        data = [self.append_buffer.popleft() for _ in range(size)]
        for i in range(size):
            priority, state, action, reward, next_state, done = \
            data[i]['priority'], data[i]['state'], data[i]['action'], data[i]['reward'], data[i]['next_state'], data[i]['done']
            for j in range(len(data[i])):
                self.buffer.add(priority[j].item(), Experience(state[j], action[j], reward[j], next_state[j], done[j], 0)) #fix log_prob
                
    def update_idxs(self):    
        size = min(len(self.update_buffer), self.max_iter)
        data = [self.update_buffer.popleft() for _ in range(size)]
        for i in range(size):
            idxs, td_errors = data[i]
            self.buffer.update_priorities(idxs, td_errors.squeeze())
@ray.remote
class ImpalaBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action, args):
        self.append_buffer = deque(maxlen = learner_memory_size)
        self.args = args
        
    def put_trajectories(self, data):
        self.append_buffer.append(data)
        
    def get_append_buffer(self):
        return self.append_buffer
    
    def sample(self):
        if len(self.append_buffer) == 0 :
            return [], 0
        size = min(len(self.append_buffer), self.args['traj_num'])
        data = [self.append_buffer.popleft() for _ in range(size)]
        state, action, reward, next_state, done, log_prob = [], [], [], [], [], [] 
        for d in (data):
            state.append(d['state'])
            action.append(d['action'])
            reward.append(d['reward'])
            next_state.append(d['next_state'])
            done.append(d['done'])
            log_prob.append(d['log_prob'])
        state, action, reward, next_state, done, log_prob = np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done), np.stack(log_prob)
        transition = make_transition(state, action, reward, next_state, done, log_prob)
        return transition, size
        