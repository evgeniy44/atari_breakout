import unittest
from unittest.mock import Mock, MagicMock

import numpy as np
from gym.spaces.discrete import Discrete

from sources.deep_q_agent import DeepQAgent
from sources.input_normalizer import InputNormalizer
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

    def test_random_action_with_epsilon_decay(self):
        np.random.seed(1)
        model_factory = ModelFactory()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(1, 7056))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model_factory.build_model(84 * 84 + 1),
                           target_network=model_factory.build_model(84 * 84 + 1),
                           epoch_length=1, epsilon_decay_frequency=2)
        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[1], np.zeros(shape=7056)))
        self.assertEquals(agent.epsilon, 1)

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[1], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[2], np.zeros(shape=7056)))
        self.assertEquals(agent.epsilon, 1)

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[1], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[2], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[3], np.zeros(shape=7056)))
        self.assertEquals(agent.epsilon, 0.98)

        self.assertEquals(agent.act(state), 3, "Should Make random action")
        self.assertTrue(np.array_equal(agent.frame[0], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[1], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[2], np.ones(shape=7056)))
        self.assertTrue(np.array_equal(agent.frame[3], np.ones(shape=7056)))

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

    def test_learn_first_steps(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        agent = DeepQAgent(action_space=Discrete(4), normalizer=InputNormalizer(dimensions=(84, 84), total_actions=4),
                           experience_size=100, model_network=model, target_network=target)
        experience_replay = Mock()
        agent.experience_replay = experience_replay

        agent.learn(state1=None, action1=1, reward=1, state2=None, done=False)
        experience_replay.observe.assert_not_called()
        model.predict.assert_not_called()
        target.predict.assert_not_called()
        target.train_on_batch.assert_not_called()
        model.train_on_batch.assert_not_called()

    def test_learn_step_frame(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        agent = DeepQAgent(action_space=Discrete(4), normalizer=InputNormalizer(dimensions=(84, 84), total_actions=4),
                           experience_size=100, model_network=model, target_network=target)
        experience_replay = Mock()
        agent.experience_replay = experience_replay
        agent.episode_step = 4
        frame_1 = np.append(np.ones(shape=(1, 84 * 84)), np.full(fill_value=2, shape=(1, 84 * 84)), axis=0)
        frame_2 = np.append(np.full(fill_value=3, shape=(1, 84 * 84)),
                            np.full(fill_value=4, shape=(1, 84 * 84)), axis=0)

        agent.frame = np.append(frame_1, frame_2, axis=0)
        agent.learn(state1=None, action1=2, reward=1, state2=None, done=False)

        self.assertTrue(np.array_equal(agent.frame, experience_replay.observe.call_args_list[0][0][0]))
        self.assertEquals(2, experience_replay.observe.call_args_list[0][0][1])
        self.assertEquals(1, experience_replay.observe.call_args_list[0][0][2])
        self.assertEquals(False, experience_replay.observe.call_args_list[0][0][3])
        model.predict.assert_not_called()
        target.predict.assert_not_called()
        target.train_on_batch.assert_not_called()
        model.train_on_batch.assert_not_called()

    def test_learn_step_frame_plus_one(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        agent = DeepQAgent(action_space=Discrete(4), normalizer=InputNormalizer(dimensions=(84, 84), total_actions=4),
                           experience_size=100, model_network=model, target_network=target)
        experience_replay = Mock()
        agent.experience_replay = experience_replay
        agent.episode_step = 5
        frame_1 = np.append(np.ones(shape=(1, 84 * 84)), np.full(fill_value=2, shape=(1, 84 * 84)), axis=0)
        frame_2 = np.append(np.full(fill_value=3, shape=(1, 84 * 84)),
                            np.full(fill_value=4, shape=(1, 84 * 84)), axis=0)

        agent.frame = np.append(frame_1, frame_2, axis=0)
        agent.learn(state1=None, action1=2, reward=1, state2=None, done=False)

        observation_1 = np.append(np.full(fill_value=2, shape=(1, 84 * 84)), np.full(fill_value=3, shape=(1, 84 * 84)),
                                  axis=0)
        observation_2 = np.append(np.full(fill_value=4, shape=(1, 84 * 84)),
                                  np.full(fill_value=1, shape=(1, 84 * 84)), axis=0)

        obs = np.append(observation_1, observation_2, axis=0)

        self.assertTrue(np.array_equal(obs, experience_replay.observe.call_args_list[0][0][0]))
        self.assertEquals(2, experience_replay.observe.call_args_list[0][0][1])
        self.assertEquals(1, experience_replay.observe.call_args_list[0][0][2])
        self.assertEquals(False, experience_replay.observe.call_args_list[0][0][3])
        model.predict.assert_not_called()
        target.predict.assert_not_called()
        target.train_on_batch.assert_not_called()
        model.train_on_batch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
