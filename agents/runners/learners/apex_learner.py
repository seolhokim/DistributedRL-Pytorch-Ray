from agents.runners.learners.learner import Learner

import ray

@ray.remote
class APEXLearner(Learner):
    def run(self, ps, buffer):
        data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
        idx, td_error = self.brain.train_network(data)
        ray.wait([ps.push.remote(self.get_weights())])
        ray.get(buffer.put_idxs.remote([idx, td_error]))
            