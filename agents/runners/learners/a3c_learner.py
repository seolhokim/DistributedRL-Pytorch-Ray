from agents.runners.learners.learner import Learner

import ray

@ray.remote
class A3CLearner(Learner):
    def apply_gradients(self, num, gradients):
        self.optimizer.zero_grad()
        self.brain.set_gradients(gradients)
        self.optimizer.step()