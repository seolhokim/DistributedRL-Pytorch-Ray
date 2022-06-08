import ray

from agents.runners.actors.actor import Actor
from utils.environment import Environment
from utils.run_env import run_env

class APEXActor(Actor):
    def __init__(self, num, algorithm, writer, device, state_dim, action_dim, agent_args, epsilon):
        self.num = num
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
        
    def run(self, env_name, ps, global_buffer, epochs):
        env = Environment(env_name)
        print("actor start")
        i = 0
        while 1:
            if i % self.args['actor_update_cycle'] == 0:
                weights = ray.get(ps.pull.remote())
                self.algorithm.set_weights(weights)
            run_env(env, self.algorithm, self.algorithm.device, self.args['traj_length'], True)
            data = self.algorithm.get_trajectories()
            td_error = self.algorithm.get_td_error(data)
            data['priority'] = td_error.detach().cpu().numpy()
            global_buffer.put_trajectories.remote(data)
            i += 1
        print('actor finish')