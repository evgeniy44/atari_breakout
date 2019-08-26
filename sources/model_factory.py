from keras import models
from keras import layers
from keras import optimizers


class ModelFactory:
    def __init__(self, learning_rate=0.0004):
        self.learning_rate = learning_rate

    def build_model(self, input_size):
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(input_size,)))  # TODO regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1))
        rms = optimizers.RMSprop(lr=self.learning_rate)
        model.compile(optimizer=rms, loss='mse', metrics=['mae'])
        return model
