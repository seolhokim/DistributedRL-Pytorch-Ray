from agents.runners.learners.learner import Learner

import ray

class A3CLearner(Learner):
    def apply_gradients(self, num, gradients, ps):
        self.optimizer.zero_grad()
        self.algorithm.set_gradients(gradients)
        self.optimizer.step()
        ray.wait([ps.push.remote(self.get_weights())])
