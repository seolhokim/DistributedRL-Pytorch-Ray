import torch.optim as optim

class Learner:
    def init(self, brain, args):
        self.brain = brain
        self.args = args
        self.optimizer = optim.Adam(self.brain.parameters(), lr = self.args['lr'])
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    

    
    