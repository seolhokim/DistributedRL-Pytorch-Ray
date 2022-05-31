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

from utils.run_env import run_env
def test_agent(env_name, agent, ps, repeat, sleep = 3):
    total_time = 0 
    while 1 :
        time.sleep(sleep)
        total_time += sleep
        agent.set_weights(ray.get(ps.pull.remote()))
        score_lst = []
        env = Environment(env_name)
        for i in range(repeat):
            score = run_env(env, agent, 'cpu')
            score_lst.append(score)
        print("time : ", total_time, "'s, ", repeat, " means performance : ", sum(score_lst)/repeat)
        if sleep == 0:
            return
        
def run(args, agent_args):
    algorithm = DPPO
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = ray.remote(num_gpus=0.1)(DPPOLearner)
    #learner = ray.remote(DPPOLearner)
    
    learner = learner.remote(algorithm, writer, device, state_dim,\
                             action_dim, agent_args)
    actors = [ray.remote(num_gpus=0.1)(DPPOActor) for _ in range(args.num_actors)]
    #actors = [ray.remote(DPPOActor) for _ in range(args.num_actors)]
    
    actors = [actor.remote(i, algorithm, writer, device, state_dim,\
                             action_dim, agent_args) for i, actor in enumerate(actors)]
    
    ps = ray.remote(num_gpus=0.1)(ParameterServer)
    #ps = ray.remote(ParameterServer)
    ps = ps.remote(ray.get(learner.get_weights.remote()))
    
    import time

    start = time.time()
    

    test_agent = ray.remote(num_gpus=0.1)(TestAgent)
    #test_agent = ray.remote(TestAgent)
    
    
    test_agent = test_agent.remote(args.env_name, algorithm, writer, device, \
                     state_dim, action_dim, agent_args, ps,\
                         repeat = 3)
    test_agent.test_agent.remote()
    
    for epoch in range(args.epochs):
        ray.wait([learner.run.remote(actors, ps, args)]) 
    print("time :", time.time() - start)


