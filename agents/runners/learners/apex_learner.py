from agents.runners.learners.learner import Learner

import ray
import time

@ray.remote
class APEXLearner(Learner):
    def run(self,buffer):
        #while 1 :
        print('learner start')
        for i in range(500):
            data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
            if len(data) > 0:
                self.brain.train_network(data)
            else :
                time.sleep(0.1)
        print('learner finish')