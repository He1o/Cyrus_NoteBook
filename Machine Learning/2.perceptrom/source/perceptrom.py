import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, train_data, lable):
        self.train_data = np.array(train_data)
        self.lable = np.array(lable)
        self.length = len(train_data)
        self.train_data = np.append(self.train_data, np.ones((self.length, 1)), axis = 1)
        self.weights = np.zeros(len(train_data[0]) + 1)
        self.eta = 0.2

    def sign(self, x):
        return 1 if x >= 0 else -1

    def predict(self, input_data):
        return self.sign(np.sum(self.weights * input_data))


    def train(self):
        for _ in range(100):
            flag = True
            for i, inpute_data in enumerate(self.train_data):
                if self.lable[i] * self.predict(inpute_data) <= 0:
                    self.weights += self.eta * self.lable[i]  * inpute_data
                    flag = False
                    plt.cla()
                    scat()
                    plt.xlim((0, 10))
                    plt.ylim((0, 10))
                    plt.plot([0,10], [(-self.weights[2] / self.weights[1]), (-self.weights[2] - 10 * self.weights[0]) / self.weights[1]])
                    plt.pause(1.5)
            if flag:
                return self.weights

train_data = [[2, 3], [1, 4], [3, 5], [2, 6], [4, 5], [3, 1], [4, 3], [6, 2], [2, 1]]
lable = [1, 1, 1, 1, 1, -1, -1, -1 ,-1]

plt.ion()


def scat():
    plt.scatter([x[0] for x in train_data[:5]], [x[1] for x in train_data[:5]], c='r')
    plt.scatter([x[0] for x in train_data[5:]], [x[1] for x in train_data[5:]], c='b')

ptron = Perceptron(train_data, lable)
ptron.train()