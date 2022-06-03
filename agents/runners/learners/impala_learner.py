from agents.runners.learners.learner import Learner

import ray
import time

class ImpalaLearner(Learner):
    def run(self, ps, buffer):
        data, size = ray.get(buffer.sample.remote())
        if size > 0 :
            self.optimizer.zero_grad()
            loss = self.algorithm.compute_gradients(data)
            loss.backward()
            self.optimizer.step()
            ray.wait([ps.push.remote(self.get_weights())])
        else :
            print("learner waits data")
            time.sleep(0.1)