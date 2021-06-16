from agents.runners.learners.learner import Learner

import ray

@ray.remote
class DPPOLearner(Learner):
    def init(self, brain, args):
        super().init(brain, args)
        self.workers = set()
        self.num_workers = self.args['num_actors']
        
    def apply_gradients(self, num, gradients):
        self.workers.add(num)
        self.brain.add_gradients(gradients)
        if len(self.workers) == self.num_workers:
            self.optimizer.step()
            self.brain.zero_grad()
            self.workers = set()
        else :
            while len(self.workers) == self.num_workers:
                time.sleep(1e-3)