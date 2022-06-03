from utils.environment import Environment

class Actor:
    def __init__(self, num, algorithm, writer, device, state_dim, action_dim, agent_args):
        self.num = num
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args).to(device)
    def get_algorithm(self):
        return self.algorithm
    
    def get_weights(self):
        return self.algorithm.get_weights()
    
    def set_weights(self, weights):
        self.algorithm.set_weights(weights)

