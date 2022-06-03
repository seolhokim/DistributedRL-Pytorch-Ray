import torch
import ray
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import TestAgent
from utils.parameter_server import ParameterServer

from agents.algorithms.dppo import DPPO
from agents.runners.learners.dppo_learner import DPPOLearner
from agents.runners.actors.dppo_actor import DPPOActor

def run(args, agent_args):
    start = time.time()
    
    algorithm = DPPO
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ray.remote(num_gpus=0.1)(DPPOLearner)
    learner = learner.remote(algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
    
    actors = [ray.remote(num_gpus=0.1)(DPPOActor) for _ in range(args.num_actors)]
    actors = [actor.remote(i, algorithm, writer, device, state_dim,\
                             action_dim, agent_args) for i, actor in enumerate(actors)]
    
    ps = ray.remote(num_gpus=0.1)(ParameterServer)
    ps = ps.remote(ray.get(learner.get_weights.remote()))
    test_agent_algorithm = DPPOActor(args.num_actors, algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
    test_agent = ray.remote(num_gpus=0.1)(TestAgent)
    test_agent = test_agent.remote(args.env_name, test_agent_algorithm, writer, device, \
                     state_dim, action_dim, agent_args, ps,\
                         repeat = 3)

    test_agent.test_agent.remote()    
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(actors, ps, args)]) 
    print("time :", time.time() - start)


