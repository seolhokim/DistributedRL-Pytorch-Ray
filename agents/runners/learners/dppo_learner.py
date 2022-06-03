import ray
from agents.runners.learners.learner import Learner

class DPPOLearner(Learner):
    def run(self, actors, ps, env_args):
        ray.wait([agent.reset.remote(env_args.env_name) for agent in actors])
        for _ in range(self.args['train_epoch']):
            weights = ray.put(self.algorithm.get_weights())
            ray.wait([agent.weight_sync.remote(weights) for agent in actors])
            gradients_ids = ([agent.run.remote() for agent in actors])
            while len(gradients_ids):
                done_id, gradients_ids = ray.wait(gradients_ids)
                self.algorithm.add_gradients(ray.get(done_id[0]))
            self.optimizer.step()
            self.algorithm.zero_grad()
            ps.push.remote(self.get_weights())