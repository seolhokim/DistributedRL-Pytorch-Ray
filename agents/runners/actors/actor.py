from utils.environment import Environment

class Actor:
    def __init__(self, num, brain, writer, device, state_dim, action_dim, agent_args):
        self.num = num
        self.args = agent_args
        self.brain = brain(writer, device, state_dim, action_dim, agent_args).to(device)
    def get_brain(self):
        return self.brain
    
    def get_weights(self):
        return self.brain.get_weights()
    
    def set_weights(self, weights):
        self.brain.set_weights(weights)

