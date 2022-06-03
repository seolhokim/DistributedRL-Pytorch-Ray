import ray
import torch
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import TestAgent
from utils.replaybuffer import ImpalaBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.a2c_vtrace import A2CVtrace
from agents.runners.actors.impala_actor import ImpalaActor
from agents.runners.learners.impala_learner import ImpalaLearner

def run(args, agent_args):
    algorithm = A2CVtrace
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ray.remote(num_gpus=0.1)(ImpalaLearner)
    learner = learner.remote(algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
    
    actors = [ray.remote(num_gpus=0.1)(ImpalaActor) for _ in range(args.num_actors)]
    actors = [actor.remote(i, algorithm, writer, device, state_dim,\
                             action_dim, agent_args) for i, actor in enumerate(actors)]
    
    ps = ray.remote(num_gpus=0.1)(ParameterServer)
    ps = ps.remote(ray.get(learner.get_weights.remote()))
    
    buffer = ImpalaBuffer.remote(agent_args['learner_memory_size'], state_dim, 1, agent_args)
    
    [actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
    test_agent_algorithm = ImpalaActor(args.num_actors, algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
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