import ray
from utils.run_env import compute_gradients

@ray.remote
class Worker:
    def init(self, brain, args):
        self.brain = brain
        self.args = args
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def train_agent(self, env_name, global_agent, epochs):
        for i in range(epochs):
            for grad in compute_gradients(env_name, global_agent, self.brain, self.args['traj_length']):
                global_agent.apply_gradients.remote(grad)
        
