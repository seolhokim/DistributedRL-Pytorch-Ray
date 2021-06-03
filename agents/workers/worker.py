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
    def set_weights(self, learner):
        weights = ray.get(learner.get_weights.remote())
        self.brain.set_weights(weights)
    def compute_gradients(self, env_name, global_agent, epochs):
        for i in range(epochs):
            for grad in run_env(env_name, global_agent, self.brain, self.args['traj_length']):
                global_agent.apply_gradients.remote(grad)