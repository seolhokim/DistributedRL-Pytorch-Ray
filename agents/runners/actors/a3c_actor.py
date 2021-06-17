import ray

from agents.runners.actors.actor import Actor
from utils.environment import Environment

@ray.remote
class A3CActor(Actor):
    def run(self, env_name, global_agent, epochs):
        env = Environment(env_name)
        for grad in self.brain.compute_gradients(env, global_agent, epochs, self.args['reward_scaling']):
            global_agent.apply_gradients.remote(self.num, grad)
