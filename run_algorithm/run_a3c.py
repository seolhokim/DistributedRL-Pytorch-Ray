import torch
import ray
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import TestAgent
from utils.parameter_server import ParameterServer

from agents.algorithms.a3c import A3C
from agents.runners.actors.a3c_actor import A3CActor
from agents.runners.learners.a3c_learner import A3CLearner

def run(args, agent_args):
    start = time.time()
    algorithm = A3C
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ray.remote(num_gpus=0.3)(A3CLearner)
    learner = learner.remote(algorithm, writer, device, state_dim,\
                             action_dim, agent_args)

    actors = [ray.remote(num_gpus=0.1)(A3CActor) for _ in range(args.num_actors)]
    actors = [actor.remote(i, algorithm, writer, device, state_dim,\
                             action_dim, agent_args) for i, actor in enumerate(actors)]

    ps = ray.remote(num_gpus=0.1)(ParameterServer)
    ps = ps.remote(ray.get(learner.get_weights.remote()))
    test_agent_algorithm = A3CActor(args.num_actors, algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
    test_agent = ray.remote(num_gpus=0.1)(TestAgent)
    test_agent = test_agent.remote(args.env_name, test_agent_algorithm, writer, device, \
                     state_dim, action_dim, agent_args, ps, repeat = 3)

    runners = [agent.run.remote(args.env_name, learner, ps, args.epochs) for agent in actors]
    test_agent.test_agent.remote() 
    while len(runners) :
        done, runners = ray.wait(runners)
    print("time :", time.time() - start)
