import gym

class Environment:
    def __init__(self,env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self._max_episode_steps = self.env._max_episode_steps
        self.can_run = False
        self.state = None
        
        if type(self.env.action_space) == gym.spaces.box.Box : #Continuous
            self.action_dim = self.env.action_space.shape[0] 
            self.is_discrete = False
        else :
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
    def reset(self):
        assert not self.can_run
        self.can_run = True
        self.state = self.env.reset()
        return self.state
    
    def step(self,action):
        assert self.can_run
        next_state, reward, done, info = self.env.step(action)
        self.state = next_state
        if done == True:
            self.can_run = False
        return next_state, reward, done, info