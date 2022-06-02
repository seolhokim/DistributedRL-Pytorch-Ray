import torch.optim as optim

class Learner:
    def __init__(self, brain, writer, device, state_dim, action_dim, agent_args):
        self.args = agent_args
        self.brain = brain(writer, device, state_dim, action_dim, agent_args).to(device)
        self.optimizer = optim.Adam(self.brain.parameters(), lr = self.args['lr'])
        
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    

    
    