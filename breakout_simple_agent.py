import agent
import matplotlib.pyplot as plt
import cv2


class SimpleBreakoutAgent(agent.Agent):

    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.9, lambda_=0.7):
        super(SimpleBreakoutAgent, self).__init__(action_space)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

    def act(self, state):


        height = 155
        width = 120
        dim = (width, height)
        res_img = cv2.resize(state, dim, interpolation=cv2.INTER_LINEAR)

        self.display_one(res_img)

        return self.action_space.sample()

    def reset(self):
        pass

    # Display one image
    def display_one(self, a, title1="Original"):
        plt.imshow(a), plt.title(title1)
        plt.xticks([]), plt.yticks([])
        plt.show()
