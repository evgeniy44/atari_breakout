import agent
import numpy as np
import cv2

from keras import models
from keras import layers
from matplotlib import pyplot as plt

from replay import ReplayMemory

INPUT_SIZE = 84 * 84 * 3 + 1


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.9, lambda_=0.7, minibatch_size=32,
                 epoch_length=250, steps_to_copy=1000):
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
        self.experience_replay = ReplayMemory(max_size=50000, observation_size=INPUT_SIZE - 1)
        self.step_counter = 0
        self.maes = []
        self.mses = []

    def act(self, state):
        if np.random.random() < self.epsilon:
            act = self.action_space.sample()
        else:
            max_value = -1000000
            ties = []
            for current_action in range(self.action_space.n):
                action_value = self.model_network.predict(self.normalize(state, current_action), batch_size=1, verbose=False)
                if np.ndarray.item(action_value) > max_value:
                    ties.clear()
                    ties.append(current_action)
                    max_value = action_value
                elif action_value == max_value:
                    ties.append(current_action)
            act = np.random.choice(ties)

        self.step_counter += 1
        return act

    def learn(self, state1, action1, reward, state2, done):
        self.experience_replay.observe(self.normalize_state(state1), action1, reward, done)

        if self.step_counter > self.epoch_length:  # 1 epoch is 100 steps
            self.current_loss = self.update_model()

    def update_model(self):
        (state, action, reward, s_next, is_terminal) = self.experience_replay.sample_minibatch(
            self.minibatch_size)  # return data from 32 steps

        s_next_predictions = np.zeros((self.minibatch_size, self.num_actions), dtype=np.float32)

        for current_action in range(self.action_space.n):
            actions = np.full(shape=(self.minibatch_size, 1, 1), fill_value=current_action).astype('float32') / 4
            input = np.reshape(np.append(s_next, actions, axis=2), (self.minibatch_size, INPUT_SIZE))
            predict = self.target_network.predict(input)
            s_next_predictions[:, current_action] = np.reshape(predict, 32)

        s_next_max_values = s_next_predictions.max(axis=1, keepdims=True)
        expected_state_values = (1 - is_terminal) * self.gamma * s_next_max_values + reward
        formatted_actions = np.reshape(action, (self.minibatch_size, 1, 1))

        current_state_and_action = np.reshape(np.append(s_next, formatted_actions, axis=2), (self.minibatch_size, INPUT_SIZE))
        # current_state_and_action = np.reshape(np.append(state, formatted_actions), (self.minibatch_size, INPUT_SIZE))

        mse, mae = self.model_network.train_on_batch(current_state_and_action, expected_state_values)

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
        flatten_state = np.reshape(resized_state, (1, INPUT_SIZE - 1))
        flatten_state = flatten_state.astype('float32') / 255
        return flatten_state

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(INPUT_SIZE,)))  # TODO regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model
