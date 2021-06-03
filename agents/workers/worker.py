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

    def compute_gradients(self, env_name, global_agent, epochs):
        for i in range(epochs):
            weights = ray.get(global_agent.get_weights.remote())
            self.brain.set_weights(weights)
            for grad in run_env(env_name, self.brain, update_interval = self.args['traj_length']):
                global_agent.apply_gradients.remote(grad)