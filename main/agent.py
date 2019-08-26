class Agent(object):

    def __init__(self, action_space):
        self.action_space = action_space
        self.num_actions = action_space.n

    def act(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError