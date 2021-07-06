import ray
import torch
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import test_agent
from utils.replaybuffer import ImpalaBuffer
from utils.parameter_server import ParameterServer


from agents.algorithms.a2c_vtrace import A2CVtrace
from agents.runners.actors.impala_actor import ImpalaActor
from agents.runners.learners.impala_learner import ImpalaLearner

def run(args, agent_args):
    algorithm = A2CVtrace
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ImpalaLearner.remote()
    actors = [ImpalaActor.remote() for _ in range(args.num_actors)]
    buffer = ImpalaBuffer.remote(agent_args['learner_memory_size'], state_dim, 1)
    
    learner.init.remote(algorithm(writer, device, state_dim, action_dim, agent_args), agent_args)
    ray.get([agent.init.remote(idx, algorithm(writer, device, state_dim, action_dim, agent_args) , agent_args) for idx, agent in enumerate(actors)])
    ps = ParameterServer.remote(ray.get(learner.get_weights.remote()))
    
    [actor.run.remote(args.env_name, ps, buffer, args.epochs) for actor in actors]
    test_agent.remote(args.env_name, algorithm(writer, device, state_dim, action_dim, agent_args), ps, args.test_repeat, args.test_sleep)
    
    time.sleep(3)
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(ps, buffer)])
    