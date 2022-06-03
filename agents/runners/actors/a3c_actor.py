import ray
from agents.runners.actors.actor import Actor
from utils.environment import Environment

class A3CActor(Actor):
    def run(self, env_name, global_agent, ps, epochs):
        env = Environment(env_name)
        for grad in self.algorithm.compute_gradients(env, ps, epochs, self.args['reward_scaling']):
            ray.wait([global_agent.apply_gradients.remote(self.num, grad, ps)])
            
