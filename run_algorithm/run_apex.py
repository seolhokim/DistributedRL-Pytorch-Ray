import ray
import torch
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import TestAgent
from utils.replaybuffer import ApexBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.dqn import DQN
from agents.runners.actors.apex_actor import APEXActor
from agents.runners.learners.apex_learner import APEXLearner

def run(args, agent_args):
    algorithm = DQN
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ray.remote(num_gpus=0.1)(APEXLearner)
    learner = learner.remote(algorithm, writer, device, state_dim,\
                             action_dim, agent_args, epsilon = 0)
    
    actors = [ray.remote(num_gpus=0.1)(APEXActor) for _ in range(args.num_actors)]
    actors = [actor.remote(i, algorithm, writer, device, state_dim,\
                             action_dim, agent_args, epsilon = (agent_args['epsilon'] ** (1 + ((i/(args.num_actors-1))* agent_args['alpha'])))) for i, actor in enumerate(actors)]
    buffer = ApexBuffer.remote(agent_args['learner_memory_size'], state_dim, 1)
    
    
    ps = ray.remote(num_gpus=0.1)(ParameterServer)
    ps = ps.remote(ray.get(learner.get_weights.remote()))
    
    [actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
    buffer_run.remote(buffer)
    test_agent_algorithm = APEXActor(args.num_actors, algorithm, writer, device, state_dim,\
                             action_dim, agent_args, epsilon = 0)
    test_agent = ray.remote(num_gpus=0.1)(TestAgent)
    test_agent = test_agent.remote(args.env_name, test_agent_algorithm, writer, device, \
                     state_dim, action_dim, agent_args, ps,\
                         repeat = 3)

    test_agent.test_agent.remote()  
    
    time.sleep(3)
    print('learner start')
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(ps, buffer)])
    print('learner finish')
     
@ray.remote
def buffer_run(buffer):
    print('buffer_start')
    while 1:
        ray.wait([buffer.stack_data.remote()])
        #synchronize issue check
        #lock it if learner added data to buffer
        ray.wait([buffer.update_idxs.remote()])
        time.sleep(0.1)
    print("buffer finished")