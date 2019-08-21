import agent
import numpy as np

from keras import models
from keras import layers

INPUT_SIZE = 210 * 160 * 3 + 1


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.9, lambda_=0.7):
        super(DeepQAgent, self).__init__(action_space)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        self.model = self.build_model()

    def act(self, state):
        if np.random.random() < self.epsilon:
            act = self.action_space.sample()
        else:
            max_value = -1
            ties = []
            for current_action in range(self.action_space.n):
                action_value = self.model.predict(self.normalize(state, current_action), batch_size=1, verbose=True)
                if np.ndarray.item(action_value) > max_value:
                    ties.clear()
                    ties.append(current_action)
                    max_value = action_value
                elif action_value == max_value:
                    ties.append(current_action)
            act = np.random.choice(ties)

        return act

    def reset(self):
        pass

    def normalize(self, state, current_action):
        flatten_state = np.reshape(state, (1, INPUT_SIZE - 1))
        flatten_state = flatten_state.astype('float32') / 255
        flatten_action = np.array(current_action)
        flatten_action = (flatten_action.astype('float32') / 255).astype('float32')

        normalized = np.append(flatten_state, flatten_action)
        return np.reshape(normalized, (1, INPUT_SIZE))

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(INPUT_SIZE,)))  # TODO regularization
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model
