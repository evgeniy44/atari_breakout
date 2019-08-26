import numpy as np

from sources import agent
from sources.replay import ReplayMemory

INPUT_SIZE = 84 * 84 * 4 + 1


class DeepQAgent(agent.Agent):

    def __init__(self, action_space, normalizer, model_network, target_network, epsilon=1, alpha=0.5, gamma=0.99,
                 lambda_=0.7, minibatch_size=32,
                 epoch_length=50000, steps_to_copy=10000, frame_size=4, experience_size=70000,
                 epsilon_decay_frequency=5000):
        super(DeepQAgent, self).__init__(action_space)

        self.epsilon_decay_frequency = epsilon_decay_frequency
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

        self.model_network = model_network
        self.target_network = target_network
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
            for current_action in range(self.num_actions):
                action_value = self.model_network.predict(self.normalizer.normalize_input(self.frame, current_action),
                                                          batch_size=1,
                                                          verbose=False)
                if np.ndarray.item(action_value) > max_value:
                    ties.clear()
                    ties.append(current_action)
                    max_value = action_value
                elif action_value == max_value:
                    ties.append(current_action)
            act = np.random.choice(ties)  # actions 0 and 1 do nothing in Atari breakout

        if self.epsilon_decay:
            if self.step_counter > self.epoch_length and self.step_counter % self.epsilon_decay_frequency == 0:
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

        s_next_predictions = np.zeros((self.minibatch_size, self.num_actions), dtype=np.float32)

        for current_action in range(self.num_actions):  # left and right
            actions = np.full(shape=(self.minibatch_size, 1), fill_value=current_action).astype('float32')
            prediction_input = np.append(s_next, actions, axis=1)
            predict = self.target_network.predict(prediction_input)
            s_next_predictions[:, current_action] = np.reshape(predict, self.minibatch_size)

        s_next_max_values = s_next_predictions.max(axis=1, keepdims=True)
        expected_state_values = (1 - np.reshape(is_terminal, (self.minibatch_size, 1)))\
                                * self.gamma * s_next_max_values + np.reshape(reward, (self.minibatch_size, 1))
        formatted_actions = np.reshape(action, (self.minibatch_size, 1))

        current_state_and_action = np.append(state, formatted_actions, axis=1)

        mse, mae = self.model_network.train_on_batch(current_state_and_action, expected_state_values)

        if self.step_counter % 1000 == 0:
            self.maes.append(mae)
            self.mses.append(mse)
            print("Step: " + str(self.step_counter) + ", mae: " + str(mae) + ", mse: " + str(mse))

        if self.step_counter % self.steps_to_copy == 0:
            self.target_network.set_weights(self.model_network.get_weights())

        return mse, mae

    def reset(self):
        self.episode_step = 0
        pass
