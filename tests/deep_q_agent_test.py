import unittest
from unittest.mock import Mock, MagicMock

import numpy as np
from gym.spaces.discrete import Discrete

from sources.deep_q_agent import DeepQAgent
from sources.model_factory import ModelFactory


class DeepQAgentTest(unittest.TestCase):
    def test_act_start(self):
        np.random.seed(1)

        model_factory = ModelFactory()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(1, 7056))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model_factory.build_model(84 * 84 + 1),
                           target_network=model_factory.build_model(84 * 84 + 1))
        state = np.random.randint(256, size=(210, 16, 3))

        action = agent.act(state)

        self.assertEquals(action, 1, "Should Start with Action 1")

        self.assertEquals(agent.episode_step, 1, "Step 4")
        self.assertEquals(agent.step_counter, 1, "Step 4")
        self.assertEquals(agent.last_action, 1, "last action")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))

    def test_random_action(self):
        np.random.seed(1)
        model_factory = ModelFactory()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(1, 7056))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model_factory.build_model(84 * 84 + 1),
                           target_network=model_factory.build_model(84 * 84 + 1))
        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertEquals(agent.act(state), 3, "Should Make random action")

        self.assertEquals(agent.episode_step, 4, "Step 4")
        self.assertEquals(agent.step_counter, 4, "Step 4")
        self.assertEquals(agent.last_action, 3, "last action")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))

    def test_model_action(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()

        model.predict = MagicMock(
            side_effect=[np.array([[0.23]]), np.array([[0.75]]), np.array([[0.11]]), np.array([[0.007]])])
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(1, 7056))
        normalizer.normalize_input.return_value = np.ones(shape=(1, 7057))

        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer, experience_size=100, model_network=model,
                           target_network=target, epsilon=0)
        agent.episode_step = 3
        agent.step_counter = 3

        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Make Action according to the model")

        self.assertEquals(agent.episode_step, 4, "Step 4")
        self.assertEquals(agent.step_counter, 4, "Step 4")
        self.assertTrue(np.array_equal(agent.frame[3], np.ones(shape=7056)))

        normalizer.normalize_state.assert_called_once_with(state)

        frame = np.zeros((4, 7056))
        frame[3, :] = np.ones(shape=(1, 7056))

        self.assertTrue(np.array_equal(frame, normalizer.normalize_input.call_args_list[0][0][0]))
        self.assertEquals(0, normalizer.normalize_input.call_args_list[0][0][1])

        self.assertTrue(np.array_equal(frame, normalizer.normalize_input.call_args_list[1][0][0]))
        self.assertEquals(1, normalizer.normalize_input.call_args_list[1][0][1])

        self.assertTrue(np.array_equal(frame, normalizer.normalize_input.call_args_list[2][0][0]))
        self.assertEquals(2, normalizer.normalize_input.call_args_list[2][0][1])

        self.assertTrue(np.array_equal(frame, normalizer.normalize_input.call_args_list[3][0][0]))
        self.assertEquals(3, normalizer.normalize_input.call_args_list[3][0][1])

        self.assertTrue(np.array_equal(np.ones(shape=(1, 7057)), model.predict.call_args_list[0][0][0]))
        self.assertTrue(np.array_equal(np.ones(shape=(1, 7057)), model.predict.call_args_list[1][0][0]))
        self.assertTrue(np.array_equal(np.ones(shape=(1, 7057)), model.predict.call_args_list[2][0][0]))
        self.assertTrue(np.array_equal(np.ones(shape=(1, 7057)), model.predict.call_args_list[3][0][0]))

        self.assertTrue(np.array_equal(frame, agent.frame))
        self.assertEquals(agent.last_action, 1)


if __name__ == '__main__':
    unittest.main()
