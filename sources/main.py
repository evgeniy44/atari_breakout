import gym

from sources.deep_q_agent import DeepQAgent
from sources.experiment import Experiment

import tensorflow as tf
from tensorflow.python.client import device_lib

from sources.input_normalizer import InputNormalizer

if not tf.test.is_gpu_available():
    raise Exception("MUST BE RUNNING ON GPU")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
device_lib.list_local_devices()
env = gym.make('Breakout-v0')
env.reset()
agent = DeepQAgent(env.action_space, InputNormalizer((84, 84)))
experiment = Experiment(env, agent)
experiment.run_it(5000000, interactive=True)

env.close()