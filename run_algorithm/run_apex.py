import ray
import torch
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import test_agent
from utils.replaybuffer import CentralizedBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.dqn import DQN
from agents.runners.actors.apex_actor import APEXActor
from agents.runners.learners.apex_learner import APEXLearner

def run(args, agent_args):
    algorithm = DQN
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = APEXLearner.remote()
    actors = [APEXActor.remote() for _ in range(args.num_actors)]
    buffer = CentralizedBuffer.remote(agent_args['learner_memory_size'], state_dim, 1)
    
    learner.init.remote(algorithm(writer, device, state_dim, action_dim, agent_args, epsilon = 0), agent_args)
    ray.get([agent.init.remote(idx, algorithm(writer, device, state_dim, action_dim, agent_args, epsilon = (agent_args['epsilon'] ** (1 + (idx/(args.num_actors-1))* agent_args['alpha']) ) ), agent_args) for idx, agent in enumerate(actors)])
    ps = ParameterServer.remote(ray.get(learner.get_weights.remote()))
    
    ray.wait([actor.fill_buffer.remote(args.env_name) for actor in actors])
    [actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
    buffer_run.remote(buffer)
    test_agent.remote(args.env_name, algorithm(writer, device, state_dim, action_dim, agent_args, epsilon = 0), ps, args.test_repeat, args.test_sleep)
    
    time.sleep(3)
    
    while 1 :
        ray.wait([learner.run.remote(ps, buffer)])
        time.sleep(agent_args['buffer_update_time'])
    
        
@ray.remote
def buffer_run(buffer):
    print('buffer_start')
    while 1:
        ray.wait([buffer.stack_data.remote()])
        ray.wait([buffer.update_idxs.remote()])
        time.sleep(0.1)
    print("buffer finished")