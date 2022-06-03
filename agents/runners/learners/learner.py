import torch.optim as optim

class Learner:
    def __init__(self, algorithm, writer, device, state_dim, action_dim, agent_args):
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args).to(device)
        self.optimizer = optim.Adam(self.algorithm.parameters(), lr = self.args['lr'])
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_weights(self):
        return self.algorithm.get_weights()
    

    
    