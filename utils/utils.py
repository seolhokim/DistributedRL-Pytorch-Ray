import torch
import numpy as np
def make_transition(state,action,reward,next_state,done):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['done'] = done
    return transition
def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
        yield [x[indices] for x in value[1:]]
def convert_to_tensor(*value):
    device = value[0]
    return [torch.tensor(x).float().to(device) for x in value[1:]]

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    #def __getattr__(self,val):
    #    return self[val]
