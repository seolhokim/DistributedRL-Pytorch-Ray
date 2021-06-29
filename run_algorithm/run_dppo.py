import torch
import ray
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import test_agent
from utils.parameter_server import ParameterServer

from agents.algorithms.dppo import DPPO
from agents.runners.learners.dppo_learner import DPPOLearner
from agents.runners.actors.dppo_actor import DPPOActor

def run(args, agent_args):
    algorithm = DPPO
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = DPPOLearner.remote()
    actors = [DPPOActor.remote() for _ in range(args.num_actors)]
    
    ray.get(learner.init.remote(algorithm(writer, device, \
                                         state_dim, action_dim, agent_args), agent_args))
    ray.get([agent.init.remote(i, algorithm(writer, device, state_dim, action_dim, agent_args), \
                       agent_args) for i, agent in enumerate(actors)])
    
    ps = ParameterServer.remote(ray.get(learner.get_weights.remote()))
    
    start = time.time()
    test_agent.remote(args.env_name, algorithm(writer, device, \
                                         state_dim, action_dim, agent_args), ps, args.test_repeat, args.test_sleep)
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(actors, ps, args)]) 
    print("time :", time.time() - start)