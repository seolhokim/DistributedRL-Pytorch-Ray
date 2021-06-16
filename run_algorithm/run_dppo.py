import torch
import ray
import time

from utils.environment import Environment
from utils.run_env import test_agent

from agents.algorithms.dppo import DPPO
from agents.runners.learners.dppo_learner import DPPOLearner
from agents.runners.actors.dppo_actor import DPPOActor

def run(args, agent_args):
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    learner = DPPOLearner.remote()
    agent_args['num_actors'] = args.num_actors
    ray.get(learner.init.remote(DPPO(writer, device, \
                                         state_dim, action_dim, agent_args), agent_args))
    actors = [DPPOActor.remote() for _ in range(args.num_actors)]
    ray.get([agent.init.remote(i, DPPO(writer, device, state_dim, action_dim, agent_args), \
                       agent_args) for i, agent in enumerate(actors)])
    
    start = time.time()
    test_agent.remote(args.env_name, learner, args.test_repeat, args.test_sleep)
    for i in range(args.epochs):
        runners = [agent.train_agent.remote(args.env_name, learner, args.epochs) for agent in actors]
        while len(runners) :
            done, runners = ray.wait(runners)  
    print("time :", time.time() - start)