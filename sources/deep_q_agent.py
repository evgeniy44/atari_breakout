import numpy as np

from sources import agent
from sources.replay import ReplayMemory
import matplotlib.pyplot as plt
import h5py
import datetime

INPUT_SIZE = 84 * 84 * 4 + 1


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, normalizer, model_network, target_network, epsilon=1, gamma=0.99,
                 minibatch_size=32, epoch_length=50000, steps_to_copy=10000, frame_size=4, experience_size=200000,
                 epsilon_decay_frequency=5000, epsilon_decay=True):
        super(DeepQAgent, self).__init__(action_space)

        self.epsilon_decay_frequency = epsilon_decay_frequency
        self.normalizer = normalizer
        self.frame_size = frame_size
        self.steps_to_copy = steps_to_copy
        self.epoch_length = epoch_length
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.frame = np.zeros((84, 84, 4), dtype=np.float32)
        self.last_action = 1
        self.episode_step = 0

        self.model_network = model_network
        self.target_network = target_network
        self.target_network.set_weights(self.model_network.get_weights())
        self.experience_replay = ReplayMemory(max_size=experience_size, observation_size=84)
        self.step_counter = 0
        self.maes = []
        self.mses = []

        self.epsilon_decay = epsilon_decay

    def act(self, state):
        self.frame[:, :, self.episode_step % self.frame_size] = self.normalizer.normalize_state(state)
        if np.random.random() < self.epsilon:
            act = np.random.randint(3) + 1
        else:
            if self.episode_step % self.frame_size == self.frame_size - 1:
                observed_state = self.frame
            else:
                observed_state = np.append(self.frame[:, :, (self.episode_step + 1) % self.frame_size:],
                                           self.frame[:, :, : (self.episode_step + 1) % self.frame_size], axis=2)
            action_values = self.model_network.predict(np.reshape(observed_state, (1, 84, 84, 4)), batch_size=1,
                                                       verbose=False)
            act = action_values.argmax()

        if self.epsilon_decay:
            if self.step_counter > self.epoch_length and self.step_counter % self.epsilon_decay_frequency == 0:
                self.epsilon = max(.01, self.epsilon * .98)

        self.last_action = act
        self.step_counter += 1
        self.episode_step += 1

        return act

    def learn(self, state1, action1, reward, state2, done):
        self.experience_replay.observe(self.normalizer.normalize_state(state1), action1, reward, done)
        if self.step_counter > self.epoch_length:
            self.current_loss = self.update_model()

    def update_model(self):
        (state, action, reward, s_next, is_terminal) = self.experience_replay.sample_minibatch(
            self.minibatch_size, frame_size=self.frame_size)  # return data from 32 steps

        next_state_action_values = self.target_network.predict(s_next)
        s_next_max_values = next_state_action_values.max(axis=1, keepdims=True)

        expected_all_state_values = self.model_network.predict(state)

        expected_best_action_state_values = (1 - is_terminal) * self.gamma * s_next_max_values + reward

        for i in range(self.minibatch_size):
            expected_all_state_values[i, int(action[i])] = expected_best_action_state_values[i]

        mse, mae = self.model_network.train_on_batch(state, expected_all_state_values)

        if self.step_counter % 5000 == 0:
            self.maes.append(mae)
            self.mses.append(mse)
            print("Step: " + str(self.step_counter) + ", mae: " + str(mae) + ", mse: " + str(mse))

        if self.step_counter % self.steps_to_copy == 0:
            self.target_network.set_weights(self.model_network.get_weights())

        if self.step_counter % 100_000:
            self.model_network.save_weights("../models/model" + datetime.datetime.now().isoformat())

        return mse, mae

    def reset(self):
        self.episode_step = 0
        pass
