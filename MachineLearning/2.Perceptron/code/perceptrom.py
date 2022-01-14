import numpy as np
import matplotlib.pyplot as plt
# plt.figure().set_size_inches(8, 8)
class Perceptron(object):
    def __init__(self, train_data, lable):
        self.data_pos = train_data
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
        for idx in range(100):
            flag = True
            for i, inpute_data in enumerate(self.train_data):
                if self.lable[i] * self.predict(inpute_data) <= 0:
                    self.vectorA = self.eta * self.lable[i] * inpute_data
                    Y = lambda x,y: (- y[2] - (x) * y[0]) / y[1]
                    plt.cla()
                    plt.figure().set_size_inches(6, 6)
                    plt.style.use('Solarize_Light2')

                    plt.subplots_adjust(top = 0.93, bottom = 0.07, right = 0.93, left = 0.07, hspace = 0, wspace = 0)
                    plt.ylim((-5, 10))
                    plt.xlim((-5, 10))               
                    plt.scatter([x[0] for x in self.data_pos[:5]], [x[1] for x in self.data_pos[:5]], c='r')
                    plt.scatter([x[0] for x in self.data_pos[5:]], [x[1] for x in self.data_pos[5:]], c='b')
                    plt.scatter(inpute_data[0], inpute_data[1], c='k')
                    plt.scatter(0, -self.weights[2] / self.weights[1], c='k')
                    plt.plot([0, self.vectorA[0]], [Y(0, self.weights), Y(0, self.weights) + self.vectorA[1]])
                    
                    plt.plot([0, self.weights[0]], [Y(0, self.weights), Y(0, self.weights) + self.weights[1]])
                    plt.plot([-5,10], [Y(-5, self.weights), Y(10, self.weights)])
                    
                    self.vectorB = self.weights + self.vectorA
                    plt.plot([0, self.vectorB[0]], [Y(0, self.weights), Y(0, self.weights) + self.vectorB[1]])
                    plt.plot([-5,10], [Y(-5, self.vectorB), Y(10, self.vectorB)])

                    print(self.weights, self.eta * self.lable[i] * inpute_data, self.predict(inpute_data), self.lable[i])
                    plt.savefig('img_{}_{}.jpg'.format(idx,i))
                    flag = False
                    
                    self.weights = self.vectorB
                    # plt.pause(0.5)
            if flag:
                return self.weights

train_data = [[2, 3], [1, 4], [3, 5], [2, 6], [4, 5], [3, 1], [4, 3], [6, 2], [2, 1]]
lable = [1, 1, 1, 1, 1, -1, -1, -1 ,-1]

plt.ion()


ptron = Perceptron(train_data, lable)
ptron.train()