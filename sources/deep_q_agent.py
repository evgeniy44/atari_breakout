from sources import agent
import numpy as np
import cv2

from keras import models
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt

from sources.replay import ReplayMemory

INPUT_SIZE = 84 * 84 * 4 + 1


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, normalizer, epsilon=1, alpha=0.5, gamma=0.99, lambda_=0.7, minibatch_size=32,
                 epoch_length=50000, steps_to_copy=10000, frame_size=4, experience_size=70000):
        super(DeepQAgent, self).__init__(action_space)

        self.normalizer = normalizer
        self.frame_size = frame_size
        self.steps_to_copy = steps_to_copy
        self.epoch_length = epoch_length
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.frame = np.zeros(self.frame_size * 1 * 84 * 84, dtype=np.float32).reshape(
            (self.frame_size, 84 * 84))
        self.frame_reward = 0
        self.last_action = 1
        self.episode_step = 0

        self.model_network = self.build_model()
        self.target_network = self.build_model()
        self.target_network.set_weights(self.model_network.get_weights())
        self.experience_replay = ReplayMemory(max_size=experience_size, observation_size=INPUT_SIZE - 1)
        self.step_counter = 0
        self.maes = []
        self.mses = []

        if self.epsilon == 1:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False

    def act(self, state):
        self.frame[self.episode_step % self.frame_size] = self.normalizer.normalize_state(state)
        if self.episode_step % self.frame_size != self.frame_size - 1:
            act = self.last_action  # do nothing
        elif np.random.random() < self.epsilon:
            act = np.random.randint(3) + 1
        else:
            max_value = -10000000
            ties = []
            for current_action in range(3):  # left or right
                action_value = self.model_network.predict(self.normalizer.normalize_input(self.frame, current_action), batch_size=1,
                                                          verbose=False)
                if np.ndarray.item(action_value) > max_value:
                    ties.clear()
                    ties.append(current_action)
                    max_value = action_value
                elif action_value == max_value:
                    ties.append(current_action)
            act = np.random.choice(ties) + 1  # actions 0 and 1 do nothing in Atari breakout

        if self.epsilon_decay:
            if self.step_counter > self.epoch_length and self.step_counter % 5000 == 0:
                self.epsilon = max(.01, self.epsilon * .98)

        self.last_action = act
        self.step_counter += 1
        self.episode_step += 1

        return act

    def learn(self, state1, action1, reward, state2, done):
        if self.episode_step >= self.frame_size:
            if self.episode_step % self.frame_size == 0:
                self.experience_replay.observe(self.frame, action1, reward, done)
            else:
                observed_state = np.append(self.frame[self.episode_step % self.frame_size:, :],
                                           self.frame[:self.episode_step % self.frame_size, :], axis=0)
                self.experience_replay.observe(observed_state, action1, reward, done)

        if self.step_counter > self.epoch_length:
            self.current_loss = self.update_model()

    def update_model(self):
        (state, action, reward, s_next, is_terminal) = self.experience_replay.sample_minibatch(
            self.minibatch_size)  # return data from 32 steps

        s_next_predictions = np.zeros((self.minibatch_size, 2), dtype=np.float32) #num actions = 2

        for current_action in range(3):  # left and right
            actions = np.full(shape=(self.minibatch_size, 1, 1), fill_value=current_action).astype('float32')
            input = np.reshape(np.append(s_next, actions, axis=2), (self.minibatch_size, INPUT_SIZE))
            predict = self.target_network.predict(input)
            s_next_predictions[:, current_action] = np.reshape(predict, self.minibatch_size)

        s_next_max_values = s_next_predictions.max(axis=1, keepdims=True)
        expected_state_values = (1 - is_terminal) * self.gamma * s_next_max_values + reward
        formatted_actions = np.reshape(action, (self.minibatch_size, 1, 1))

        current_state_and_action = np.reshape(np.append(s_next, formatted_actions, axis=2),
                                              (self.minibatch_size, INPUT_SIZE))
        # current_state_and_action = np.reshape(np.append(state, formatted_actions), (self.minibatch_size, INPUT_SIZE))

        mse, mae = self.model_network.train_on_batch(current_state_and_action, expected_state_values)

        if self.step_counter % 1000 == 0:
            self.maes.append(mae)
            self.mses.append(mse)
            print("Step: " + str(self.step_counter) + ", mae: " + str(mae) + ", mse: " + str(mse))

        if self.step_counter % self.steps_to_copy == 0:
            self.target_network.set_weights(self.model_network.get_weights())

        return mse, mae

    def plot_training_stats(self):
        # Plot the episode length over time
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(self.mses)
        plt.xlabel("Steps / 50")
        plt.ylabel("MSE")
        plt.title("MSE over time")

        # Plot the episode reward over time
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(self.maes)
        plt.xlabel("Steps / 50")
        plt.ylabel("MAE")
        plt.title("MAE over Time")
        plt.show()

        return fig1, fig2

    def reset(self):
        self.episode_step = 0
        pass

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(INPUT_SIZE,)))  # TODO regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1))
        rms = optimizers.RMSprop(lr=0.0004)
        model.compile(optimizer=rms, loss='mse', metrics=['mae'])
        return model
