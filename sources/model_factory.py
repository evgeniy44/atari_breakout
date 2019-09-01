from keras import models
from keras import layers
from keras import optimizers


class ModelFactory:
    def __init__(self, learning_rate=0.00025):
        self.learning_rate = learning_rate

    def build_model(self, weights_file):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), activation='relu', strides=4, input_shape=(84, 84, 4)))
        model.add(layers.Conv2D(64, (4, 4), activation='relu', strides=2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', strides=1))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(4))

        rms = optimizers.RMSprop(lr=self.learning_rate)
        model.compile(optimizer=rms, loss='mse', metrics=['mae'])
        if weights_file is not None:
            model.load_weights(weights_file)

        model.summary()
        return model
