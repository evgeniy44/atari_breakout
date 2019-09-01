import unittest
from unittest.mock import Mock, MagicMock

import numpy as np
from gym.spaces.discrete import Discrete

from sources.deep_q_agent import DeepQAgent
from sources.input_normalizer import InputNormalizer


class DeepQAgentTest(unittest.TestCase):
    def test_act_start(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()

        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model,
                           target_network=target)
        state = np.random.randint(256, size=(210, 16, 3))

        action = agent.act(state)

        self.assertEquals(action, 3, "Should Start with Action 1")

        self.assertEquals(agent.episode_step, 1, "Step 4")
        self.assertEquals(agent.step_counter, 1, "Step 4")
        self.assertEquals(agent.last_action, 3, "last action")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))

    def test_random_action_with_epsilon_decay(self):
        np.random.seed(1)
        normalizer = Mock()
        model = MagicMock()
        target = MagicMock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model,
                           target_network=target,
                           epoch_length=1, epsilon_decay_frequency=2)
        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 3, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 1], np.zeros(shape=(84, 84))))
        self.assertEquals(agent.epsilon, 1)

        self.assertEquals(agent.act(state), 3, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 1], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 2], np.zeros(shape=(84, 84))))
        self.assertEquals(agent.epsilon, 1)

        self.assertEquals(agent.act(state), 1, "Should Start with Action 1")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 1], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 2], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 3], np.zeros(shape=(84, 84))))
        self.assertEquals(agent.epsilon, 0.98)

        self.assertEquals(agent.act(state), 3, "Should Make random action")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 1], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 2], np.ones(shape=(84, 84))))
        self.assertTrue(np.array_equal(agent.frame[:, :, 3], np.ones(shape=(84, 84))))

        self.assertEquals(agent.episode_step, 4, "Step 4")
        self.assertEquals(agent.step_counter, 4, "Step 4")
        self.assertEquals(agent.last_action, 3, "last action")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))

    def test_model_action(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()

        model.predict = MagicMock(
            side_effect=[np.array([[0.23, 0.75, 0.11, 0.007]])])
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))

        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer, experience_size=100, model_network=model,
                           target_network=target, epsilon=0)
        agent.episode_step = 3
        agent.step_counter = 3

        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Make Action according to the model")

        self.assertEquals(agent.episode_step, 4, "Step 4")
        self.assertEquals(agent.step_counter, 4, "Step 4")
        self.assertTrue(np.array_equal(agent.frame[:, :, 3], np.ones(shape=(84, 84))))

        normalizer.normalize_state.assert_called_once_with(state)

        frame = np.zeros((84, 84, 4))
        frame[:, :, 3] = np.ones(shape=(84, 84))

        self.assertTrue(np.array_equal(frame, model.predict.call_args_list[0][0][0][0]))

        self.assertTrue(np.array_equal(frame, agent.frame))
        self.assertEquals(agent.last_action, 1)

    def test_model_action(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()

        model.predict = MagicMock(
            side_effect=[np.array([[0.23, 0.75, 0.11, 0.007]])])
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))

        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer, experience_size=100, model_network=model,
                           target_network=target, epsilon=0)
        agent.episode_step = 4
        agent.step_counter = 4

        state = np.random.randint(256, size=(210, 16, 3))

        self.assertEquals(agent.act(state), 1, "Should Make Action according to the model")

        self.assertEquals(agent.episode_step, 5, "Step 4")
        self.assertEquals(agent.step_counter, 5, "Step 4")
        self.assertTrue(np.array_equal(agent.frame[:, :, 0], np.ones(shape=(84, 84))))

        normalizer.normalize_state.assert_called_once_with(state)

        frame = np.zeros((84, 84, 4))
        frame[:, :, 0] = np.ones(shape=(84, 84))

        self.assertTrue(np.array_equal(np.append(agent.frame[:,:,1:], agent.frame[:,:,:1], axis=2), model.predict.call_args_list[0][0][0][0]))

        self.assertTrue(np.array_equal(frame, agent.frame))
        self.assertEquals(agent.last_action, 1)

    def test_learn_first_step(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model, target_network=target)
        experience_replay = Mock()
        agent.experience_replay = experience_replay

        agent.learn(state1=np.zeros((84, 84)), action1=2, reward=1, state2=None, done=False)

        self.assertTrue(np.array_equal(np.ones((84, 84)), experience_replay.observe.call_args_list[0][0][0]))
        self.assertEquals(2, experience_replay.observe.call_args_list[0][0][1])
        self.assertEquals(1, experience_replay.observe.call_args_list[0][0][2])
        self.assertEquals(False, experience_replay.observe.call_args_list[0][0][3])
        model.predict.assert_not_called()
        target.predict.assert_not_called()
        target.train_on_batch.assert_not_called()
        model.train_on_batch.assert_not_called()

    def test_learn_step_frame_and_update_model(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model, target_network=target, minibatch_size=4, gamma=0.1)
        experience_replay = Mock()
        agent.experience_replay = experience_replay
        agent.episode_step = 50004
        agent.step_counter = 50004

        target.predict = MagicMock(
            side_effect=[np.array([[0.12, 0.13, 0.14, 0.15],
                                   [0.06, 0.75, 0.27, 0.28],
                                   [0.01, 0.01, 0.98, 0.22],
                                   [0.02, 0.07, 0.06, 0.89]])])

        current_action_values = np.array(
            [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.1, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]])

        model.predict = MagicMock(
            side_effect=[current_action_values])

        states = np.full(fill_value=0.5, shape=(4, 84, 84, 4))
        next_states = np.full(fill_value=0.7, shape=(4, 84, 84, 4))
        actions = np.array([0, 1, 2, 3])
        rewards = np.array([[0], [0], [1], [0]])
        is_terminal = np.array([[False], [False], [False], [True]])
        experience_replay.sample_minibatch.return_value = (states, actions, rewards, next_states, is_terminal)
        model.train_on_batch.return_value = (0.5, 0.25)

        agent.learn(state1=np.zeros((84, 84)), action1=2, reward=1, state2=None, done=False)

        expected_training_action_values = np.array(
            [[0.15 * agent.gamma, 0.02, 0.03, 0.04], [0.05, 0.75 * agent.gamma, 0.07, 0.08],
             [0.09, 0.1, 0.98 * agent.gamma + 1, 0.12], [0.13, 0.14, 0.15, 0]])

        self.assertTrue(np.array_equal(states, model.train_on_batch.call_args_list[0][0][0]))
        self.assertTrue(
            np.array_equal(
                np.round_(expected_training_action_values, decimals=4),
                np.round_(model.train_on_batch.call_args_list[0][0][1], decimals=4)))

        self.assertTrue(np.array_equal(np.ones(shape=(84, 84)), experience_replay.observe.call_args_list[0][0][0]))
        self.assertEquals(2, experience_replay.observe.call_args_list[0][0][1])
        self.assertEquals(1, experience_replay.observe.call_args_list[0][0][2])
        self.assertEquals(False, experience_replay.observe.call_args_list[0][0][3])

        self.assertTrue(np.array_equal(states, model.predict.call_args_list[0][0][0]))
        target.train_on_batch.assert_not_called()
        model.get_weights.assert_called_once()
        target.set_weights.assert_called_once()

    def test_learn_step_frame_and_update_model_and_copy_weights(self):
        np.random.seed(1)

        model = MagicMock()
        target = MagicMock()
        normalizer = Mock()
        normalizer.normalize_state.return_value = np.ones(shape=(84, 84))
        agent = DeepQAgent(action_space=Discrete(4), normalizer=normalizer,
                           experience_size=100, model_network=model, target_network=target, minibatch_size=4, gamma=0.1)
        experience_replay = Mock()
        agent.experience_replay = experience_replay
        agent.episode_step = 60000
        agent.step_counter = 60000

        target.predict = MagicMock(
            side_effect=[np.array([[0.12, 0.13, 0.14, 0.15],
                                   [0.06, 0.75, 0.27, 0.28],
                                   [0.01, 0.01, 0.98, 0.22],
                                   [0.02, 0.07, 0.06, 0.89]])])

        current_action_values = np.array(
            [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.1, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]])

        model.predict = MagicMock(
            side_effect=[current_action_values])

        states = np.full(fill_value=0.5, shape=(4, 84, 84, 4))
        next_states = np.full(fill_value=0.7, shape=(4, 84, 84, 4))
        actions = np.array([0, 1, 2, 3])
        rewards = np.array([[0], [0], [1], [0]])
        is_terminal = np.array([[False], [False], [False], [True]])
        experience_replay.sample_minibatch.return_value = (states, actions, rewards, next_states, is_terminal)
        model.train_on_batch.return_value = (0.5, 0.25)

        agent.learn(state1=np.zeros((84, 84)), action1=2, reward=1, state2=None, done=False)

        expected_training_action_values = np.array(
            [[0.15 * agent.gamma, 0.02, 0.03, 0.04], [0.05, 0.75 * agent.gamma, 0.07, 0.08],
             [0.09, 0.1, 0.98 * agent.gamma + 1, 0.12], [0.13, 0.14, 0.15, 0]])

        self.assertTrue(np.array_equal(states, model.train_on_batch.call_args_list[0][0][0]))
        self.assertTrue(
            np.array_equal(
                np.round_(expected_training_action_values, decimals=4),
                np.round_(model.train_on_batch.call_args_list[0][0][1], decimals=4)))

        self.assertTrue(np.array_equal(np.ones(shape=(84, 84)), experience_replay.observe.call_args_list[0][0][0]))
        self.assertEquals(2, experience_replay.observe.call_args_list[0][0][1])
        self.assertEquals(1, experience_replay.observe.call_args_list[0][0][2])
        self.assertEquals(False, experience_replay.observe.call_args_list[0][0][3])

        self.assertTrue(np.array_equal(states, model.predict.call_args_list[0][0][0]))
        target.train_on_batch.assert_not_called()

        self.assertEquals(model.get_weights.call_count, 2)
        self.assertEquals(target.set_weights.call_count, 2)


if __name__ == '__main__':
    unittest.main()
