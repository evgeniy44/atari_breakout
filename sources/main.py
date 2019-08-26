import gym

from sources.deep_q_agent import DeepQAgent
from sources.experiment import Experiment

import tensorflow as tf
from tensorflow.python.client import device_lib

from sources.input_normalizer import InputNormalizer
from sources.model_factory import ModelFactory

if not tf.test.is_gpu_available():
    raise Exception("MUST BE RUNNING ON GPU")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
device_lib.list_local_devices()
env = gym.make('Breakout-v0')
env.reset()
model_factory = ModelFactory()
dimensions = (84, 84)
agent = DeepQAgent(env.action_space, InputNormalizer(dimensions, total_actions=4),
                   model_network=model_factory.build_model(4 * dimensions[0] * dimensions[1] + 1),
                   target_network=model_factory.build_model(4 * dimensions[0] * dimensions[1] + 1))
experiment = Experiment(env, agent)
experiment.run_it(5000000, interactive=True)

env.close()
