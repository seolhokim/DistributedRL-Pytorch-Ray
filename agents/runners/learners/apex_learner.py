from agents.runners.learners.learner import Learner

import ray

class APEXLearner(Learner):
    def __init__(self, algorithm, writer, device, state_dim, action_dim, agent_args, epsilon):
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
    def run(self, ps, buffer):
        data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
        idx, td_error = self.algorithm.train_network(data)
        ray.wait([ps.push.remote(self.get_weights())])
        ray.wait([buffer.put_idxs.remote([idx, td_error])])