import torch
import ray
import time

from utils.utils import run_setting
from utils.environment import Environment
from utils.run_env import test_agent

from agents.algorithms.a3c import A3C
from agents.runners.actors.a3c_actor import A3CActor
from agents.runners.learners.a3c_learner import A3CLearner

def run(args, agent_args):
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    learner = A3CLearner.remote()
    ray.get(learner.init.remote(A3C(writer, device, \
                                         state_dim, action_dim, agent_args), agent_args))
    actors = [A3CActor.remote() for _ in range(args.num_actors)]
    ray.get([agent.init.remote(i, A3C(writer, device, state_dim, action_dim, agent_args), \
                       agent_args) for i, agent in enumerate(actors)])
    
    start = time.time()
    test_agent.remote(args.env_name, learner, args.test_repeat, args.test_sleep)

    runners = [agent.train_agent.remote(args.env_name, learner, args.epochs) for agent in actors]
    while len(runners) :
        done, runners = ray.wait(runners)
    
    print("time :", time.time() - start)
