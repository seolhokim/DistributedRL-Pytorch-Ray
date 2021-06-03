import ray
from utils.utils import run_env

@ray.remote
class Worker:
    def init(self, brain, args):
        self.brain = brain
        self.args = args
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()

    def compute_gradients(self, env_name, global_agent):
        weights = ray.get(global_agent.get_weights.remote())
        self.brain.set_weights(weights)
        for grad in run_env(env_name, self.brain, train = True, update_interval = self.args['update_interval']):
            global_agent.apply_gradients.remote(grad)