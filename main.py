import gym
import time
from breakout_simple_agent import SimpleBreakoutAgent
from experiment import Experiment

env = gym.make('Breakout-v0')
env.reset()
agent = SimpleBreakoutAgent(env.action_space)
experiment = Experiment(env, agent)
experiment.run_simple(5, interactive=True)


# env.reset()
# for _ in range(1000):
#     env.render()
#
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
#     time.sleep(0.1)
env.close()