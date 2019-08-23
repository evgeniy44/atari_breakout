import gym

from breakout_simple_agent import SimpleBreakoutAgent
from deep_q_agent import DeepQAgent
from experiment import Experiment

import tensorflow as tf
from tensorflow.python.client import device_lib

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

env = gym.make('Breakout-v0')
env.reset()
agent = DeepQAgent(env.action_space)
experiment = Experiment(env, agent)
experiment.run_it(5000000, interactive=True)

env.close()