import ray

import torch.optim as optim
@ray.remote
class Learner:
    def init(self, brain, args):
        
        self.brain = brain
        self.args = args
        self.optimizer = optim.Adam(self.brain.parameters(), lr = self.args['lr'])
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def apply_gradients(self, gradients):
        self.optimizer.zero_grad()
        self.brain.set_gradients(gradients)
        self.optimizer.step()