import ray
import time

import torch.optim as optim
@ray.remote
class Learner:
    def init(self, num_workers, brain, args):
        self.num_workers = num_workers
        self.brain = brain
        self.args = args
        self.optimizer = optim.Adam(self.brain.parameters(), lr = self.args['lr'])
        self.workers = set()
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def apply_gradients(self, num, gradients):
        algo = self.brain.name()  
        if algo == 'a3c':
            self.optimizer.zero_grad()
            self.brain.set_gradients(gradients)
            self.optimizer.step()
        elif algo == 'dppo':
            self.workers.add(num)
            self.brain.add_gradients(gradients)
            if len(self.workers) == self.num_workers:
                self.optimizer.step()
                self.brain.zero_grad()
                self.workers = set()
            else :
                while len(self.workers) == self.num_workers:
                    time.sleep(1e-3)
    
    