import ray
from utils.environment import Environment

class Actor:
    def init(self, num, brain, args):
        self.num = num
        self.brain = brain
        self.args = args
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    

     
            
