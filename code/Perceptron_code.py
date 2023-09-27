from typing import List
import random
import csv
import matplotlib.pyplot as plt

import numpy


class Perceptron:
    def __init__(self, eta: float = 0.8, max_iter: int = 300):
        """

        :param eta: (0,1.0]
        :param max_iter: the max iteration time
        """
        self.eta = eta
        self.max_iter = max_iter
        random.seed(1)

    def fit(self, train: List[List[float]], test: List[int]) -> None:
        if len(train) <= 0:
            return
        # get the random value for the weight and bias
        self.w = [random.randint(-10, 10) for self.w in range(len(train[0]) + 1)]
        time = 0
        while time < self.max_iter:
            # the time cannot more than the max_inter
            time += 1
            errors = 0
            for x, y in zip(train, test):
                # get the result by count the weight
                y_predict = self.predict(x)
                # if the result is different from truly outcome, then change the weight and bias
                if y != y_predict:
                    errors += 1
                    for i in range(len(x)):
                        # update the weights
                        self.w[i + 1] += self.eta * (y - y_predict) * x[i]
                    # update the bias
                    self.w[0] += self.eta * (y - y_predict)
                # print('times: {}, x: {}, y: {}, y_predict: {}, w: {}'.format(time, x, y, y_predict, self.w))
            if errors == 0:
                break
        # print(self.w)

    def predict(self, xi: List[float]) -> int:
        return 1 if sum([self.w[i + 1] * xi[i] for i in range(len(xi))]) + self.w[0] >= 0 else -1

if '__main__' == __name__:
    # set the eta and max_iter
    p = Perceptron(eta = 1.0, max_iter=1000)
    # training
    x_trainer = []
    y_trainer = []
    x_test = []
    y_test = []
    # with open('new_data.csv') as f:
    with open('diabetes.csv') as f:
        reader = csv.reader(f)
        next(reader)
        trainer = list(reader)[1:500]
        for sample in trainer:
            # first 8 elements as the working part, last outcome is the truly result
            x_trainer.append([float(i) for i in sample[:8]])
            y_trainer.append(1 if sample[8] == '1' else -1)
        # print(y_trainer)
        p.fit(x_trainer, y_trainer)

    # predict
    # with open('new_data.csv') as f:
    with open('diabetes.csv') as f:
        reader2 = csv.reader(f)
        next(reader2)
        test = list(reader2)[500:]
        correct_number = 0
        total_number = 0
        for contect in test:
            x_test.append([float(i) for i in contect[:8]])
            y_test.append(1 if contect[8] == '1' else -1)
        for i in range(len(x_test)):
            total_number += 1
            if p.predict(x_test[i]) == y_test[i]:
                correct_number += 1
        print(correct_number)
        print(total_number)
        print(correct_number/total_number)

        # # etas = list()
        # max_iters = list()
        # result = list()
        # # for eta in numpy.arange(0, 1, 0.05):
        # for max_iter in numpy.arange(100, 1001, 50):
        #     max_iters.append(max_iter)
        #     # p = Perceptron(eta, max_iter=1000)
        #     p = Perceptron(eta = 0.45, max_iter = max_iter)
        #     # training
        #     x_trainer = []
        #     y_trainer = []
        #     x_test = []
        #     y_test = []
        #     # with open('new_data.csv') as f:
        #     with open('diabetes.csv') as f:
        #         reader = csv.reader(f)
        #         next(reader)
        #         trainer = list(reader)[1:500]
        #         for sample in trainer:
        #             x_trainer.append([float(i) for i in sample[:8]])
        #             y_trainer.append(1 if sample[8] == '1' else -1)
        #         p.fit(x_trainer, y_trainer)
        #
        #     # predict
        #     # with open('new_data.csv') as f:
        #     with open('diabetes.csv') as f:
        #         reader2 = csv.reader(f)
        #         next(reader2)
        #         test = list(reader2)[500:]
        #         correct_number = 0
        #         total_number = 0
        #         for contect in test:
        #             x_test.append([float(i) for i in contect[:8]])
        #             y_test.append(1 if contect[8] == '1' else -1)
        #         for i in range(len(x_test)):
        #             total_number += 1
        #             if p.predict(x_test[i]) == y_test[i]:
        #                 correct_number += 1
        #         # print(correct_number / total_number)
        #         result.append(correct_number / total_number)
        # # print(etas)
        # print(max_iters)
        # print(result)
        # # plt.plot(etas, result)
        # # plt.legend()
        # # plt.show()