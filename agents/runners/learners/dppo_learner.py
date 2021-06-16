from agents.runners.learners.learner import Learner

import ray

@ray.remote
class DPPOLearner(Learner):
    def init(self, brain, args):
        super().init(brain, args)
        self.actors = set()
        self.num_actors = self.args['num_actors']
        
    def apply_gradients(self, num, gradients):
        self.actors.add(num)
        self.brain.add_gradients(gradients)
        if len(self.actors) == self.num_actors:
            self.optimizer.step()
            self.brain.zero_grad()
            self.workers = set()
        else :
            while len(self.actors) == self.num_actors:
                time.sleep(1e-3)