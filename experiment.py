import numpy as np
import plotting as plotting

class Experiment(object):
    def __init__(self, env, agent):

        self.env = env
        self.agent = agent

        self.episode_length = np.array([])
        self.episode_reward = np.array([])

    def run_simple(self, max_number_of_episodes=100, interactive=False):

        # repeat for each episode
        for episode_number in range(max_number_of_episodes):

            # initialize state
            state = self.env.reset()
            self.agent.reset()

            done = False  # used to indicate terminal state
            R = 0  # used to display accumulated rewards for an episode
            t = 0  # used to display accumulated steps for an episode i.e episode length

            # choose action from state using policy derived from Q
            action = self.agent.act(state)

            # repeat for each step of episode, until state is terminal
            while not done:

                t += 1  # increase step counter - for display

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # choose next action from next state using policy derived from Q
                next_action = self.agent.act(next_state)
                action = next_action

                R += reward  # accumulate reward - for display

                # if interactive display, show update for each step
                if interactive:
                    self.env.render()

            self.episode_length = np.append(self.episode_length, t)  # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward, R)  # keep episode reward - for display

        plotting.plot_episode_stats(self.episode_length, self.episode_reward)


def update_display_step(self):
    if not hasattr(self, 'imgplot'):
        self.imgplot = self.ax.imshow(self.env.render(mode='rgb_array'), interpolation='none', cmap='viridis')
    else:
        self.imgplot.set_data(self.env.render(mode='rgb_array'))

    self.fig.canvas.draw()


def update_display_episode(self):
    self.line.set_data(range(len(self.episode_length)), self.episode_length)
    self.ax1.set_xlim(0, max(10, len(self.episode_length) + 1))
    self.ax1.set_ylim(0, max(self.episode_length) + 1)

    self.line2.set_data(range(len(self.episode_reward)), self.episode_reward)
    self.ax2.set_xlim(0, max(10, len(self.episode_reward) + 1))
    self.ax2.set_ylim(min(self.episode_reward) - 1, max(self.episode_reward) + 1)

    self.fig.canvas.draw()


def run_qlearning(self, max_number_of_episodes=100, interactive=False, display_frequency=1):
    # repeat for each episode
    for episode_number in range(max_number_of_episodes):

        # initialize state
        state = self.env.reset()

        done = False  # used to indicate terminal state
        R = 0  # used to display accumulated rewards for an episode
        t = 0  # used to display accumulated steps for an episode i.e episode length

        # repeat for each step of episode, until state is terminal
        while not done:

            t += 1  # increase step counter - for display

            # choose action from state using policy derived from Q
            action = self.agent.act(state)

            # take action, observe reward and next state
            next_state, reward, done, _ = self.env.step(action)

            # agent learn (Q-Learning update)
            self.agent.learn(state, action, reward, next_state, done)

            # state <- next state
            state = next_state

            R += reward  # accumulate reward - for display

            # if interactive display, show update for each step
            if interactive:
                self.update_display_step()

        self.episode_length = np.append(self.episode_length, t)  # keep episode length - for display
        self.episode_reward = np.append(self.episode_reward, R)  # keep episode reward - for display

        # if interactive display, show update for the episode
        # if interactive:
        #     self.update_display_episode()

    # if not interactive display, show graph at the end
    if interactive:
        self.fig.clf()
        stats = plotting.EpisodeStats(
            episode_lengths=self.episode_length,
            episode_rewards=self.episode_reward,
            episode_running_variance=np.zeros(max_number_of_episodes))
        plotting.plot_episode_stats(stats, display_frequency)


def run_sarsa(self, max_number_of_episodes=100, interactive=False, display_frequency=1):
    # repeat for each episode
    for episode_number in range(max_number_of_episodes):

        # initialize state
        state = self.env.reset()
        self.agent.reset()

        done = False  # used to indicate terminal state
        R = 0  # used to display accumulated rewards for an episode
        t = 0  # used to display accumulated steps for an episode i.e episode length

        # choose action from state using policy derived from Q
        action = self.agent.act(state)

        # repeat for each step of episode, until state is terminal
        while not done:

            t += 1  # increase step counter - for display

            # take action, observe reward and next state
            next_state, reward, done, _ = self.env.step(action)

            # choose next action from next state using policy derived from Q
            next_action = self.agent.act(next_state)

            # agent learn (SARSA update)
            # self.agent.learn(state, action, reward, next_state, next_action)

            # state <- next state, action <- next_action
            state = next_state
            action = next_action

            R += reward  # accumulate reward - for display

            # if interactive display, show update for each step
            if interactive:
                self.update_display_step()

        self.episode_length = np.append(self.episode_length, t)  # keep episode length - for display
        self.episode_reward = np.append(self.episode_reward, R)  # keep episode reward - for display

        # if interactive display, show update for the episode
        if interactive:
            self.update_display_episode()

    # if not interactive display, show graph at the end
    if not interactive:
        self.fig.clf()
        stats = plotting.EpisodeStats(
            episode_lengths=self.episode_length,
            episode_rewards=self.episode_reward,
            episode_running_variance=np.zeros(max_number_of_episodes))
        plotting.plot_episode_stats(stats, display_frequency)
