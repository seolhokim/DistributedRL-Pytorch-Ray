from agents.runners.learners.learner import Learner

import torch.optim as optim
import ray

class APEXLearner(Learner):
    def __init__(self, brain, writer, device, state_dim, action_dim, agent_args, epsilon):
        self.args = agent_args
        self.brain = brain(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr = self.args['lr'])
    def run(self, ps, buffer):
        data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
        idx, td_error = self.brain.train_network(data)
        ray.wait([ps.push.remote(self.get_weights())])
        ray.get(buffer.put_idxs.remote([idx, td_error]))
            