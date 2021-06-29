from agents.runners.learners.learner import Learner
from utils.run_env import test_agent

import ray

@ray.remote
class DPPOLearner(Learner):
    def run(self, actors, env_args):
        ray.wait([agent.reset.remote(env_args.env_name) for agent in actors])
        for _ in range(self.args['train_epoch']):
            weights = ray.put(self.brain.get_weights())
            ray.wait([agent.weight_sync.remote(weights) for agent in actors])
            gradients_ids = ([agent.run.remote() for agent in actors])
            while len(gradients_ids):
                done_id, gradients_ids = ray.wait(gradients_ids)
                self.brain.add_gradients(ray.get(done_id[0]))
            self.optimizer.step()
            self.brain.zero_grad()