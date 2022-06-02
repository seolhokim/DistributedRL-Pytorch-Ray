import ray

from agents.runners.actors.actor import Actor
from utils.environment import Environment
from utils.run_env import run_env

class APEXActor(Actor):
    def __init__(self, num, brain, writer, device, state_dim, action_dim, agent_args, epsilon):
        self.num = num
        self.args = agent_args
        self.brain = brain(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
        
    def run(self, env_name, ps, global_buffer, epochs):
        env = Environment(env_name)
        print("actor start")
        i = 0
        while 1:
        #for j in range():
            if i % self.args['actor_update_cycle'] == 0:
                weights = ray.get(ps.pull.remote())
                self.brain.set_weights(weights)
            run_env(env, self.brain, self.brain.device, self.args['traj_length'], True)
            data = self.brain.get_trajectories()
            td_error = self.brain.get_td_error(data)
            data['priority'] = td_error.detach().numpy()
            global_buffer.put_trajectories.remote(data)
            i += 1
        print('actor finish')