from agents.runners.learners.learner import Learner

import ray
import time

@ray.remote
class APEXLearner(Learner):
    def run(self, ps, buffer):
        #while 1 :
        print('learner start')
        for i in range(self.args['train_epoch']):
            data = ray.get(buffer.sample.remote())
            self.brain.train_network(data)
            ray.wait([ps.push.remote(self.get_weights())])
            ray.get(buffer.put_idxs.remote([idx, td_error]))
        print('learner finish')