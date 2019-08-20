import agent
import numpy as np

class SimpleBreakoutAgent(agent.Agent):

    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.9, lambda_=0.7):
        super(SimpleBreakoutAgent, self).__init__(action_space)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

    def act(self, state):
        return self.action_space.sample()

    def reset(self):
        pass
