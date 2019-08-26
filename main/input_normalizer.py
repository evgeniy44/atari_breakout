import cv2
import numpy as np


class InputNormalizer:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def normalize_state(self, state):
        resized_state = cv2.resize(self.rgb2gray(state), self.dimensions, interpolation=cv2.INTER_LINEAR)
        flatten_state = np.reshape(resized_state, (1, self.dimensions[0] * self.dimensions[1]))
        flatten_state = flatten_state.astype('float32') / 255
        return flatten_state