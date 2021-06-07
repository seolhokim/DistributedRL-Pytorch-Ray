import ray
from utils.environment import Environment

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
        env = Environment(env_name)
        for i in range(epochs):
            for grad in self.brain.compute_gradients(env, global_agent, self.args['traj_length'], self.args['reward_scaling']):
                global_agent.apply_gradients.remote(grad)
        
