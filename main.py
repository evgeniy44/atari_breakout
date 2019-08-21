import gym

from breakout_simple_agent import SimpleBreakoutAgent
from deep_q_agent import DeepQAgent
from experiment import Experiment

env = gym.make('Breakout-v0')
env.reset()
agent = DeepQAgent(env.action_space)
experiment = Experiment(env, agent)
experiment.run_it(5000, interactive=True)


# env.reset()
# for _ in range(1000):
#     env.render()
#
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
#     time.sleep(0.1)
env.close()