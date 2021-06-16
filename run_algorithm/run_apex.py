import ray
import torch
import time

from utils.environment import Environment
from utils.run_env import test_agent
from utils.replaybuffer import CentralizedBuffer

from agents.algorithms.dqn import DQN
from agents.runners.actors.apex_actor import APEXActor
from agents.runners.learners.apex_learner import APEXLearner

def run(args, agent_args):
    args, agent_args, env, state_dim, action_dim, writer, device = run_setting(args, agent_args)
    
    learner = APEXLearner.remote()
    agent_args['learner'] = True
    learner.init.remote(DQN(device, state_dim, action_dim, agent_args), agent_args)

    actors = [APEXActor.remote() for _ in range(args.num_actors)]

    buffer = CentralizedBuffer.remote(agent_args['learner_memory_size'], state_dim, 1) 
    agent_args['learner'] = False
    ray.get([agent.init.remote(idx, DQN(device, state_dim, action_dim, agent_args), agent_args) for idx, agent in enumerate(actors)])

    [actor.run.remote(args.env_name, learner, buffer, args.epochs) for actor in actors]
    test_agent.remote(args.env_name, learner, args.test_repeat, args.test_sleep)
    buffer_run.remote(buffer)

    time.sleep(1)
    while 1 :
        learner.run.remote(buffer)
        #brain = ray.get(learner.get_brain.remote())
        #print(next(brain.actor.parameters())[0])
        time.sleep(3)
        
@ray.remote
def buffer_run(buffer):
    print('buffer_start')
    while 1:
    #for i in range(500):
        ray.get(buffer.stack_data.remote())
        time.sleep(0.1)
    print("buffer finished")