from agents.runners.actors.actor import Actor
from utils.environment import Environment

class DPPOActor(Actor):
    def reset(self, env_name):
        env = Environment(env_name)
        self.algorithm.reset(env, reward_scaling = 0.1)
        
    def weight_sync(self,weights):
        self.algorithm.set_weights(weights)
    
    def run(self):
        return self.algorithm.compute_gradients()