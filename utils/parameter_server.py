import ray

@ray.remote
class ParameterServer:
    def __init__(self, weights):
        self.weights = weights
        
    def push(self, weights):
        self.weights = weights
        
    def pull(self):
        return self.weights