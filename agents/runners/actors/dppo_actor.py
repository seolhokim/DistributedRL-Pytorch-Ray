import ray

from agents.runners.actors.actor import Actor
from utils.environment import Environment

@ray.remote
class DPPOActor(Actor):
    def reset(self, env_name):
        env = Environment(env_name)
        self.brain.reset(env, reward_scaling = 0.1)
        
    def weight_sync(self,weights):
        self.brain.set_weights(weights)
    
    def run(self):
        return self.brain.compute_gradients()