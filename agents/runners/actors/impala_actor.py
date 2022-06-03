import ray

from agents.runners.actors.actor import Actor
from utils.environment import Environment
from utils.run_env import run_env

class ImpalaActor(Actor):
    def run(self, env_name, ps, global_buffer, epochs):
        env = Environment(env_name)
        print("actor start")
        i = 0
        while 1:
        #for j in range():
            weights = ray.get(ps.pull.remote())
            self.algorithm.set_weights(weights)
            run_env(env, self.algorithm, self.algorithm.device, self.args['traj_length'], True)
            data = self.algorithm.get_trajectories()
            global_buffer.put_trajectories.remote(data)
        print('actor finish')