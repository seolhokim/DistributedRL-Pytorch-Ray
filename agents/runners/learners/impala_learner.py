from agents.runners.learners.learner import Learner

import ray
import time

@ray.remote
class ImpalaLearner(Learner):
    def run(self, ps, buffer):
        print('learner start')
        data = ray.get(buffer.sample.remote())
        if len(data) > 0 :
            self.optimizer.zero_grad()
            loss = self.brain.compute_gradients(data)
            loss.backward()
            self.optimizer.step()
            ray.wait([ps.push.remote(self.get_weights())])
        else :
            print("learner wait data")
            time.sleep(0.1)
        print('learner finish')