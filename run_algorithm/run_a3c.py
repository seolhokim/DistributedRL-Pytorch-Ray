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

    test_agent = ray.remote(num_gpus=0.1)(TestAgent)
    test_agent = test_agent.remote(args.env_name, algorithm, writer, device, \
                     state_dim, action_dim, agent_args, ps, repeat = 3)
    try : 
        runners = [agent.run.remote(args.env_name, learner, ps, args.epochs) for agent in actors]
    except Exception as e:
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
    test_agent.test_agent.remote() 
    while len(runners) :
        done, runners = ray.wait(runners)

    time.sleep(5)
    print("time :", time.time() - start)
