from agents.runners.learners.learner import Learner

import ray
import time

@ray.remote
class APEXLearner(Learner):
    def run(self,buffer):
        #while 1 :
        print('learner start')
        for i in range(self.args['train_epoch']):
            data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
            self.brain.train_network(data)
        print('learner finish')