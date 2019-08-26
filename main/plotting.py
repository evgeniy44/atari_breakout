from collections import namedtuple
from matplotlib import pyplot as plt


def plot_episode_stats(episode_lengths, episode_rewards):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.show()

    return fig1, fig2