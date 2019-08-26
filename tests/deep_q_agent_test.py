import unittest

from gym.spaces.discrete import Discrete
import numpy as np

from main.deep_q_agent import DeepQAgent


class DeepQAgentTest(unittest.TestCase):
    def test_act_start(self):
        np.random.seed(1)

        agent = DeepQAgent(action_space=Discrete(4), experience_size=100)
        state = np.random.randint(256, size=(210, 16, 3))

        action = agent.act(state)

        self.assertEquals(action, 1, "Should Start with Action 1")

    def test_random_action(self):
        np.random.seed(1)

        agent = DeepQAgent(action_space=Discrete(4), experience_size=100)
        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 3, "Should Make random action")

        self.assertEquals(agent.episode_step, 4, "Step 4")
        self.assertEquals(agent.step_counter, 4, "Step 4")
        # self.assertEquals(agent.frame[0], ) # TODO check frame


if __name__ == '__main__':
    unittest.main()
