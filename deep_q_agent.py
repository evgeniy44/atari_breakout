import agent
import numpy as np
import cv2

from keras import models
from keras import layers
from matplotlib import pyplot as plt

from replay import ReplayMemory

INPUT_SIZE = 84 * 84 * 3


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, epsilon=1, alpha=0.5, gamma=0.9, lambda_=0.7, minibatch_size=32,
                 epoch_length=50000, steps_to_copy=1000):
        super(DeepQAgent, self).__init__(action_space)

        self.steps_to_copy = steps_to_copy
        self.epoch_length = epoch_length
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        self.model_network = self.build_model()
        self.target_network = self.build_model()
        self.target_network.set_weights(self.model_network.get_weights())
        self.experience_replay = ReplayMemory(max_size=50000, observation_size=INPUT_SIZE)
        self.step_counter = 0
        self.maes = []
        self.mses = []

        if self.epsilon == 1:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False

    def act(self, state):
        if np.random.random() < self.epsilon:
            act = self.action_space.sample()
        else:

            action_values = self.model_network.predict(self.normalize_state(state), batch_size=1, verbose=False)
            act = action_values.argmax()

        if self.epsilon_decay:
            if self.step_counter % 5000 == 0:
                self.epsilon = max(.01, self.epsilon * .95)

        self.step_counter += 1


        return act

    def learn(self, state1, action1, reward, state2, done):
        self.experience_replay.observe(self.normalize_state(state1), action1, reward, done)

        if self.step_counter > self.epoch_length:  # 1 epoch is 100 steps
            self.current_loss = self.update_model()

    def update_model(self):
        (state, action, reward, s_next, is_terminal) = self.experience_replay.sample_minibatch(
            self.minibatch_size)  # return data from 32 steps

        next_state_action_values = self.target_network.predict(np.reshape(s_next, (self.minibatch_size, INPUT_SIZE)))
        s_next_max_values = next_state_action_values.max(axis=1, keepdims=True)

        current_state_values = self.model_network.predict(np.reshape(state, (self.minibatch_size, INPUT_SIZE)))

        expected_state_values = (1 - is_terminal) * self.gamma * s_next_max_values + reward

        for i in range(self.minibatch_size):
            current_state_values[i, action[i]] = expected_state_values[i]

        mse, mae = self.model_network.train_on_batch(np.reshape(state, (self.minibatch_size, INPUT_SIZE)), current_state_values)

        if self.step_counter % 50 == 0:
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
        pass

    def normalize(self, state, current_action):
        flatten_state = self.normalize_state(state)
        flatten_action = np.array(current_action)
        flatten_action = (flatten_action.astype('float32') / 4).astype('float32')

        normalized = np.append(flatten_state, flatten_action)
        return np.reshape(normalized, (1, INPUT_SIZE))

    def normalize_state(self, state):
        dim = (84, 84)
        resized_state = cv2.resize(state, dim, interpolation=cv2.INTER_LINEAR)
        flatten_state = np.reshape(resized_state, (1, INPUT_SIZE))
        flatten_state = flatten_state.astype('float32') / 255
        return flatten_state

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(INPUT_SIZE,)))  # TODO regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(4))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model
